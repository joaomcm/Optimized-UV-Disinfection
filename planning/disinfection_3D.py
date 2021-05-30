"""
    Copyright 2021 University of Illinois Board of Trustees. All Rights Reserved.

    Licensed under the “Non-exclusive Research Use License” (the "License");

    The License is included in the distribution as LICENSE.txt file.

    See the License for the specific language governing permissions and imitations under the License.

"""


# starting out with imports
import os
import sys
import pickle
import trimesh as tm
import pyrender
import numpy as np
import time
import random
import math
import klampt
import networkx as nx
import open3d as o3d
import gurobipy as gp
from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import pathlib

from klampt import IKObjective
from klampt import WorldModel,Geometry3D
from klampt import vis
from klampt.math import so3,se3
from klampt.model.create import moving_base_robot
from klampt.io import resource
from klampt.math import vectorops,so3
from klampt.model.collide import WorldCollider
from klampt.model import ik
from klampt.plan.cspace import CSpace,MotionPlan
from klampt.model.trajectory import RobotTrajectory,Trajectory
from klampt.math.vectorops import interpolate
from klampt.plan.cspace import CSpace,MotionPlan
from klampt.vis.glprogram import GLProgram
from klampt.plan import cspace,robotplanning


from importlib import reload
from visibility import Visibility
from joblib import Parallel,delayed
from tqdm import tqdm
from scipy.sparse import lil_matrix

from gurobipy import GRB
from joblib import Parallel,delayed
from sklearn.neighbors import KNeighborsClassifier
from planning.getToursAndPaths import getTour, readTourFile, getPathFromPrm, getFinalPath
from planning.robot_cspaces import Robot_3D_CSpace,CSpaceObstacleSolver

vis.init('PyQt')

# total_dofs = 11,linear_dofs = [0,1,2],angular_dofs = [4,5,6,7,8],frozen_dofs = [2,9],base_height_link = 2
# active_dofs = [0,1,4,5,6,7,8,9]

class Disinfection_Problem:
    def __init__(self,total_dofs,
    linear_dofs,
    angular_dofs,
    frozen_dofs,
    base_height_link,
    robot_height,
    lamp_linknum,
    lamp_local_coords,
    active_dofs,
    robot_cspace_generator,
    robot_cspace_solver,
    float_height,
    initial_points = 1000
    ):
        self.total_dofs = total_dofs
        self.linear_dofs = linear_dofs
        self.angular_dofs = angular_dofs
        self.frozen_dofs = frozen_dofs
        self.robot_height = robot_height
        self.lamp_linknum = lamp_linknum
        self.base_height_link = base_height_link
        self.local_lamp_coords = lamp_local_coords
        self.active_dofs = active_dofs
        self.robot_cspace_generator = robot_cspace_generator
        self.robot_cspace_solver_generator = robot_cspace_solver
        self.float_height = float_height
        self.initial_points = initial_points
        pass
    
    def get_sampling_places(self,mesh_file,resolution = 5000,robot_height = 1.5,convex_scale = 0.9):
        mesh = tm.load(mesh_file)

        # We then gimport open3d as o3d
        bounds = tm.bounds.corners(mesh.bounding_box_oriented.bounds)

        # resolution = (np.max(bounds)-np.min(bounds))/divs
        resolution = resolution/1000
        x_ = np.arange(bounds[:,0].min() - resolution,bounds[:,0].max()+resolution,resolution)
        y_ = np.arange(bounds[:,1].min() - resolution,bounds[:,1].max()+resolution,resolution)
        z_ = np.arange(0,robot_height+resolution,resolution)

        x,y,z = np.meshgrid(x_, y_, z_, indexing='ij')
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        sampling_places = np.zeros(shape = (z.shape[0],3))
        sampling_places[:,0] = x
        sampling_places[:,1] = y
        sampling_places[:,2] = z

        # we then get the convex hull of the mesh and shrink it to at least sample points that are closer to being "inside"
        this_mesh = tm.load_mesh(mesh_file)
        # we then uniformly scale the convex hull by 0.8

        # we first center the mesh:
        bounds = tm.bounds.corners(this_mesh.bounding_box_oriented.bounds)

        max_bounds = bounds.max(axis = 0)
        min_bounds = bounds.min(axis = 0)
        mean = (max_bounds-min_bounds)/2

        original_max_bounds = deepcopy(max_bounds)
        max_bounds = convex_scale*(max_bounds - mean)+mean
        min_bounds = convex_scale*(min_bounds - mean)+mean

        min_x = min_bounds[0]
        max_x = max_bounds[0]
        min_y = min_bounds[1]
        max_y = max_bounds[1]

        sampling_places = sampling_places[np.logical_and(sampling_places[:,0]>min_x,sampling_places[:,0]<max_x)]
        sampling_places = sampling_places[np.logical_and(sampling_places[:,1]>min_y,sampling_places[:,1]<max_y)]

        return sampling_places,resolution,original_max_bounds
    
    def get_irradiance_matrix(self,vis_tester,mesh,sampling_places):

        
        total_faces = mesh.mesh.faces.shape[0]
        irradiance_matrix = lil_matrix((sampling_places.shape[0],total_faces))
        for i in tqdm(range(sampling_places.shape[0])):
            _,irradiance = vis_tester.render(id0 =None,id1 = None,pos = sampling_places[i,:].tolist())
            irradiance = irradiance
            irradiance_matrix[i,np.where(irradiance > 0)] = irradiance[irradiance > 0]

        return irradiance_matrix.tocsr()

    def set_robot_link_collision_margins(self,robot,margin,collider,range_adjust_collision):
        for link_num in range_adjust_collision:
            this_link = robot.link(link_num)
            this_link.geometry().setCollisionMargin(margin)
            collider.ignoreCollision((robot.link(link_num),robot.link(link_num+1)))

    def setup_robot_and_light(self,robotfile = './data/armbot.rob',
                                    mesh_file = './full_detail_hospital_cad_meters.obj',float_height = 0.08):
        world = WorldModel()
        res1 = world.loadElement(robotfile)
        robot = world.robot(0)
        #world.loadElement(robotfile)

        #a = Geometry3D()
        # res = a.loadFile(mesh_file)
        res = world.loadElement(mesh_file)
        print(res)
        collider = WorldCollider(world)
        #we then dilate the base nd ignore collisions between it and the 2 subsequent links:
        collider.ignoreCollision((robot.link(self.base_height_link),robot.link(3)))
        collider.ignoreCollision((robot.link(self.base_height_link),robot.link(3)))
        collider.ignoreCollision((robot.link(8),robot.link(6)))
        # we now 
        cfig = robot.getConfig()
        terrain = world.terrain(0)
        lamp = robot.link(self.lamp_linknum)
        print('\n\n\nbase height link = {}, lamp linknum = {}\n\n\n'.format(self.base_height_link,self.lamp_linknum))
        cfig[self.base_height_link] = float_height
        robot.setConfig(cfig)
        robot.link(self.lamp_linknum).appearance().setColor(210/255,128/255,240/255,1)

        world.saveFile('disinfection.xml')
        return world,robot,lamp,collider


    def collisionChecker(self,collider):
        if(list(collider.collisions())!= []):
            # print('\n\n\n New Collision \n\n')
            # for i in collider.collisions():
            #     print(i[0].getName(),i[1].getName())
            # print(list(collider.collisions()))
            return False

        return True
    
    def solve_ik_near_sample(self,robot,lamp,collider,world,place,restarts = 10,tol = 1e-3,neighborhood = 0.4,float_height = 0.08):
        goal = place.tolist()
        obj = ik.objective(lamp,local = self.local_lamp_coords, world = goal)
        solver = ik.solver(obj)
        solver.setMaxIters(100)
        solver.setTolerance(tol)
        jl = solver.getJointLimits()
        jl[0][0] = goal[0] - neighborhood
        jl[1][0] = goal[0] + neighborhood
        jl[0][2] = float_height
        jl[0][1] = goal[1] - neighborhood
        jl[1][1] = goal[1] + neighborhood
        jl[1][2] = float_height
        solver.setJointLimits(jl[0],jl[1])
        solver.setActiveDofs(self.active_dofs)
        for i in range(restarts):
            if(solver.solve()):
                if(self.collisionChecker(collider)):
                    return True
                else:
                    solver.sampleInitial()
            else:
                solver.sampleInitial()
        return False

    def determine_reachable_points_robot(self,sampling_places,world,robot,lamp,collider,show_vis = False,neighborhood = 0.4,float_height = 0.08,base_linknum = 2):
        # myrob = resource.get('/home/motion/Klampt-examples/data/robots/tx90ball.rob')

        # before determining the reachable points, let us expand the robot to encourage more "spread out" configurations
    
        self.set_robot_link_collision_margins(robot,0.03,collider,self.angular_dofs)
        base = robot.link(base_linknum)
        base.geometry().setCollisionMargin(0.05)
        
        if(show_vis):
            vis.add('world',world)
            vis.show()
        
        reachable = []
        configs = []
        for place in tqdm(sampling_places):
            # time.sleep(0.2)
            reachable.append(self.solve_ik_near_sample(robot,lamp,collider,world,place,restarts = 10,tol = 1e-2,neighborhood = neighborhood,float_height = float_height))
            # print('reachable? = {}'.format(reachable[-1]))
            cfig = robot.getConfig()
        #         print(cfig[2])
            configs.append(cfig+place.tolist())
        
        # after checking for those margins, we reset the robot to its original configs for general motion planning
        self.set_robot_link_collision_margins(robot,0,collider,self.angular_dofs)
        # we also reset the base and environment collision to simplify path planning:
        world.terrain(0).geometry().setCollisionMargin(0)
        base = robot.link(self.base_height_link)
        base.geometry().setCollisionMargin(0)
        self.set_robot_link_collision_margins(robot,0,collider,self.angular_dofs)


    
        return reachable,configs

    def find_distances(self,original_max_bounds,robot,
        collider,lamp,milestones):
        space = self.robot_cspace_generator(original_max_bounds,robot,collider,lamp,milestones,
            base_height_link = self.base_height_link,
            robot_height = self.robot_height,
            float_height = self.float_height,
            linear_dofs = self.linear_dofs,
            angular_dofs = self.angular_dofs,
            light_local_position  = self.local_lamp_coords)
        program = self.robot_cspace_solver_generator(space,milestones = milestones, initial_points= self.initial_points,steps = 200)
        adjacency_matrix,roadmap,node_coords = program.get_adjacency_matrix_from_milestones()
        program.planner.space.close()
        program.planner.close()
        return adjacency_matrix,roadmap,node_coords

    def solve_model_limited_time(self,visible_irradiance,min_fluence = 280,power = 80, max_time = 2,visible_area_weights = []):
        if(len(visible_area_weights) == 0):
            visible_area_weights = np.zeros(visible_irradiance.shape[1])
            visible_area_weights[:] = 1
        print(visible_irradiance)
        
        m = gp.Model('Irradiance Solver LP')
        # m.params.Method = 3

        # we create the time variables:
        times = m.addMVar(shape = visible_irradiance.shape[0],vtype = GRB.CONTINUOUS, name = 'times')

        # and the corresponding slack variables for each face:
        face_slacks = m.addMVar(shape = visible_irradiance.shape[1], vtype = GRB.CONTINUOUS, name = 'face slacks')

        m.setObjective(times.sum() + (1000*visible_area_weights@face_slacks),GRB.MINIMIZE)
        zeros_times = np.zeros(times.shape[0])
        zeros_slacks = np.zeros(face_slacks.shape[0])
        min_fluence_vector = min_fluence*1.1*np.ones(visible_irradiance.shape[1])
        m.addConstr((power*visible_irradiance.transpose()@times + face_slacks) >= min_fluence_vector)
        m.addConstr(times >= zeros_times)
        m.addConstr(face_slacks >= zeros_slacks)
        m.addConstr(times.sum() <= 60*60*max_time)
        m.update()
        m.optimize()
        return times.x

    def perform_experiment(self,results_dir = './3D_results',mesh_file = './remeshed_hospital_room_full_70k.obj',from_scratch = True,irradiance_from_scratch = True,
        min_distance = 0.05,
        float_height = 0.08,
        resolution = 500,
        tmax = 0.5,
        convex_scale = 0.9,
        min_fluence = 280,
        power  = 80, 
        experiment = '30_mins',
        robot_name = 'armbot'):
        
        
        mesh_path =  pathlib.PurePath(mesh_file)

        mesh_name = mesh_path.name
        results_dir = results_dir+'/{}'.format(mesh_name)
        if(not os.path.exists(results_dir)):
            os.makedirs(results_dir,exist_ok = True)

        results_dir = results_dir + '/{}'.format(experiment)
        if(not os.path.exists(results_dir)):
            os.makedirs(results_dir,exist_ok = True)

        results_dir = results_dir + '/{}'.format(robot_name)
        if(not os.path.exists(results_dir)):
            os.makedirs(results_dir,exist_ok = True)


        pcd_file = results_dir + "/{}_used_points_{}_divs.pcd".format(robot_name,resolution)
        reachable_points_file = results_dir + '/{}_reachable_{}_divs.p'.format(robot_name,resolution)
        irradiance_file = results_dir + '/{}_irradiance_matrix_{}_divs.p'.format(robot_name,resolution)
        sampling_places_file = results_dir + '/{}_sampling_places_{}_divs.p'.format(robot_name,resolution)
        configs_file = results_dir + '/{}_configs_{}_divs.p'.format(robot_name,resolution)
        solutions_file = results_dir + '/{}_solutions_{}_divs.p'.format(robot_name,resolution)
        adjacency_file = results_dir + '/{}_adjacency_{}_divs.p'.format(robot_name,resolution)
        roadmap_file = results_dir + '/{}_roadmap_{}_divs.p'.format(robot_name,resolution)
        node_coords_file = results_dir + '/{}_node_coords_{}_divs.p'.format(robot_name,resolution)

        mesh = tm.load(mesh_file)
        centroid = mesh.centroid

        sampling_places,grid_resolution,original_max_bounds = self.get_sampling_places(mesh_file, resolution = resolution,convex_scale = convex_scale)



 

        pickle.dump(sampling_places,open(sampling_places_file,'wb'))
        # we then determine the robot's reachability:


        # we filter out all the sampling places that are too close to obstacles:
        # sampling_places = sampling_places[sdf > min_distance]
        discretized_z = 50


        vis_tester =Visibility(mesh_file,res = 512, useShader = True,createWnd = True)
        m = vis_tester.mesh
        self.robot_name = robot_name



        final_irradiances = []




        world,robot,lamp,collider = self.setup_robot_and_light(mesh_file = mesh_file,float_height = float_height)

        # vis.add('world',world)
        # vis.show()
        # time.sleep(5)
        if(from_scratch):
            start = time.time()
            reachable,configs = self.determine_reachable_points_robot(sampling_places,world,robot,lamp,collider,show_vis = False,float_height = float_height)
            print('\n\n computing reachability took: {} seconds \n\n'.format(time.time()-start))
            pickle.dump(reachable,open(reachable_points_file,'wb'))
            pickle.dump(configs,open(configs_file,'wb'))


            # # if this is the first time I run with this settings:

            #     vis_tester =Visibility(m,res = 512, useShader = True)
            if(irradiance_from_scratch):
                irradiance_matrix = self.get_irradiance_matrix(vis_tester,m,sampling_places[reachable])
                pickle.dump(irradiance_matrix,open(irradiance_file,'wb'))



        #else, just load everything from file:
        else:
            irradiance_matrix = pickle.load(open(irradiance_file,'rb'))
            reachable = pickle.load(open(reachable_points_file,'rb'))
            configs = pickle.load(open(configs_file,'rb'))
            # we then filter out the unreachable candidate points and calculate the irradiance matrix for all other points
        # sampling_places = sampling_places[reachable]
        configs = np.array(configs).tolist()


        mu_single = 60*60*power*0.5*irradiance_matrix
        print("\n\n number of reachable points : {} \n".format(np.sum(reachable)))
        vis.clear()
        time.sleep(1)
        vis.add('world',world)
        vis.show()

        ## We then process the optimal irradiation spots:

        #process irradiation spots

        areas = m.area()
        visible_points = np.asarray(((irradiance_matrix.sum(axis = 0)) > 0)).flatten()
        visible_areas = areas[visible_points]

        total_visible_areas = visible_areas.sum()
        total_visible_areas


        # irradiance_matrix = irradiance_matrix[reachable]

        visible_points = np.asarray(((irradiance_matrix.sum(axis = 0)) > 0)).flatten()

        visible_irradiance = irradiance_matrix[:,visible_points]

        visible_areas = areas[visible_points]
        # we then define the visible area weights:



        visible_area_weights = np.exp(100*(visible_areas/visible_areas.sum()))
        visible_area_weights[:] = 1

        # building the LP:

        
        if(from_scratch):
            solutions = []
            time_allowances = [tmax]
            for max_time in time_allowances:
                m = self.solve_model_limited_time(visible_irradiance,min_fluence,power,tmax,visible_area_weights = visible_area_weights)
                solutions.append(m)
            pickle.dump(solutions,open(solutions_file,'wb'))

        else:
            solutions = pickle.load(open(solutions_file,'rb'))

        # irradiance_matrix = pickle.load(open('irradiance_matrix_10_divs_cpu.p','rb'))
        # irradiance_matrix[irradiance_matrix < 0.001] = 0
        visible_points = np.asarray(((irradiance_matrix.sum(axis = 0)) > 0)).flatten()
        visible_irradiance = irradiance_matrix[:,visible_points]



        # we now calculate the euclidean distance matrices for all the non-zero points:
        times = solutions[0]
        total_time = times.sum()
        points_mask = times>0.5
        used_points = sampling_places[reachable,:][points_mask,:]
        print(used_points.shape)

        mus = power*np.array(np.matmul(visible_irradiance.todense().transpose(),times)).flatten()
        coverage = visible_areas[mus > min_fluence].sum()
        print('Total Environment Coverage = {:.2f}m^2'.format(coverage))

        ## Turning usable points into a point cloud

        #we then transform the grid into a point cloud


        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(used_points[:, :3])
        # o3d.visualization.draw_geometries([pcd])

        o3d.io.write_point_cloud(pcd_file, pcd)

        reachable_configs = np.array(configs)[reachable,:]
        milestones = reachable_configs[points_mask,:]


        ## We now create a prm to find the distances between the points to calculate the TSP


        t0 = time.time()

        # vis.kill()
        # from_scratch = True
        if(from_scratch):
            adjacency_matrix,roadmap,node_coords = self.find_distances(original_max_bounds,robot,collider,lamp,milestones)
            pickle.dump(adjacency_matrix,open(adjacency_file,'wb'))
            pickle.dump(roadmap,open(roadmap_file,'wb'))
            pickle.dump(node_coords,open(node_coords_file,'wb'))

        else:
            adjacency_matrix = pickle.load(open(adjacency_file,'rb'))
            roadmap = pickle.load(open(roadmap_file,'rb'))
            node_coords = pickle.load(open(node_coords_file,'rb'))

        ## Turning usable points into a point cloud

        #we then transform the grid into a point cloud


        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(used_points[:, :3])
        # o3d.visualization.draw_geometries([pcd])

        o3d.io.write_point_cloud(pcd_file, pcd)

        ## Calculating TSP


        # we then solve the TSP:
        distances = np.zeros(shape = (adjacency_matrix.shape[0]+1,adjacency_matrix.shape[0]+1))
        distances[1:,1:] = 100*adjacency_matrix.copy()

        getTour(distances, '/{}_currTSP'.format(self.robot_name)) # We just have an arbitrary name since it doesn't matter - can change this so that user can input filename if desired
        tour = readTourFile(os.path.abspath('./{}_currTSP.txt'.format(self.robot_name)), used_points)
        tour = (np.array(tour[1:])-1).tolist()
        indices = np.array(range(sampling_places[reachable,:].shape[0]))
        used_indices = indices[points_mask]
        tour_indices = used_indices[tour]

        # We then calculate the total distance travelled - and compute final trajectory:
        total_distance = 0 
        final_trajectory = []
        for i in range(len(tour)-1):
            current_point = tour[i]
            next_point = tour[i+1]
            total_distance += adjacency_matrix[current_point,next_point]
            nodes_path = nx.algorithms.shortest_path(roadmap,source = current_point,target = next_point,weight = 'weight')
        #     print(nodes_path)
        #     print(nodes_path)
            traj = [node_coords[j][:self.total_dofs+1] for j in nodes_path]
        #     print(trajectory)
            final_trajectory.append(traj[:-1])
        #     final_trajectory.append(pathDict[current_point,next_point])
        print("Total Distance Travelled by the EE = {} | Total Time Spent in Transit = {} ".format(total_distance,total_distance/30))

        return total_distance,coverage,grid_resolution,results_dir


