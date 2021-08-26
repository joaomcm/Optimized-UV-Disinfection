import pandas as pd
from tqdm import tqdm
from klampt.model.collide import WorldCollider
from klampt import WorldModel,Geometry3D

from planning.disinfection3d import DisinfectionProblem
from planning.robot_cspaces import Robot3DCSpace,CSpaceObstacleSolver 
import pdb
from klampt import vis
import time
import numpy as np
import pickle
import trimesh as tm
from copy import deepcopy
from scipy.sparse import lil_matrix,csr_matrix


class TowerbotDisinfectionProblem(DisinfectionProblem):
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
        super().__init__(total_dofs,
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
        float_height,initial_points)
    def setup_robot_and_light(self,robotfile = './data/towerbot.rob',
                                    mesh_file = './full_detail_hospital_cad_meters.obj',float_height = 0.08):
        world = WorldModel()
        res1 = world.loadElement(robotfile)
        robot = world.robot(0)
        qmin,qmax = robot.getJointLimits()
        qmin = np.nan_to_num(qmin,posinf = 1000,neginf =-100)
        qmax = np.nan_to_num(qmax,posinf = 1000,neginf =-100)
        # qmin[3] = -np.pi
        # qmax[3] = np.pi
        for index in self.frozen_dofs:
            qmin[index] = 0
            qmax[index] = 0

        robot.setJointLimits(qmin,qmax)
        #world.loadElement(robotfile)

        #a = Geometry3D()
        # res = a.loadFile(mesh_file)
        res = world.loadElement(mesh_file)
        print(res)
        collider = WorldCollider(world)
        #we then dilate the base nd ignore collisions between it and the 2 subsequent links:
        # collider.ignoreCollision((robot.link(self.base_height_link),robot.link(3)))
        # collider.ignoreCollision((robot.link(self.base_height_link),robot.link(3)))
        # collider.ignoreCollision((robot.link(8),robot.links(6)))
        # we now 
        cfig = robot.getConfig()
        terrain = world.terrain(0)
        lamp = robot.link(self.lamp_linknum)
        print('\n\n\nbase height link = {}, lamp linknum = {}\n\n\n'.format(self.base_height_link,self.lamp_linknum))
        cfig[self.base_height_link] = self.float_height
        robot.setConfig(cfig)
        robot.link(3).appearance().setColor(210/255,128/255,240/255,1)

        world.saveFile('disinfection.xml')
        return world,robot,lamp,collider

    def set_robot_link_collision_margins(self,robot,margin,collider):
        for link_num in range(3):
            this_link = robot.link(link_num)
            this_link.geometry().setCollisionMargin(margin)

    def get_sampling_places(self,mesh_file,resolution = 5000,robot_height = 1.5,convex_scale = 0.9):
        mesh = tm.load(mesh_file)
        robot_height = 1.9
        # transform resolution to meters 
        resolution = resolution/1000
        # We then gimport open3d as o3d
        bounds = tm.bounds.corners(mesh.bounding_box_oriented.bounds)

        # resolution = (np.max(bounds)-np.min(bounds))/divs
        # print(resolution)

        x_ = np.arange(bounds[:,0].min() - resolution,bounds[:,0].max()+resolution,resolution)
        y_ = np.arange(bounds[:,1].min() - resolution,bounds[:,1].max()+resolution,resolution)
        x,y = np.meshgrid(x_, y_, indexing='ij')
        x = x.flatten()
        y = y.flatten()
        sampling_places = np.zeros(shape = (x.shape[0],2))
        sampling_places[:,0] = x
        sampling_places[:,1] = y
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

        # we then create the third axis with the precise float height:
        new_sampling_places = np.zeros(shape = (sampling_places.shape[0],3))
        new_sampling_places[:,:2] = sampling_places
        new_sampling_places[:,2] = self.float_height


        return new_sampling_places,resolution,original_max_bounds

    def get_irradiance_matrix(self,vis_tester,mesh,sampling_places):

        discretized_z = 10
        z_samples = np.linspace(0.4,1.55,discretized_z)

        irradiances = []
        time.sleep(3)
        for place in tqdm(sampling_places):
            this_query = np.zeros(shape = (z_samples.shape[0],3))
            this_query[:,:2] = place[:2]
            this_query[:,2] = z_samples 
            # print(this_query)
            a = self.get_irradiance_row(vis_tester,vis_tester.mesh,this_query).sum(axis = 0)/discretized_z
            irradiances.append(a)

        irradiance_matrix = lil_matrix(np.array(irradiances))
        irradiance_matrix = irradiance_matrix.reshape(-1,irradiance_matrix.shape[-1])
        print('IRRADIANCE MATRIX SHAPE = {}'.format(irradiance_matrix.shape))

        return irradiance_matrix.tocsr()

    def get_irradiance_row(self,vis_tester,mesh,sampling_places):

        # print('dies here?')
        total_faces = mesh.mesh.faces.shape[0]
        tmp_irradiance_matrix = lil_matrix((sampling_places.shape[0],total_faces))
        for i in range(sampling_places.shape[0]):
            # print('or here?')
            _,irradiance = vis_tester.render(id0 =None,id1 = None,pos = sampling_places[i,:].tolist())
            irradiance = irradiance
            # print('or here 2!?')
            tmp_irradiance_matrix[i,np.where(irradiance > 0)] = irradiance[irradiance > 0]
            # print('it cannot seriously die here?')

        return tmp_irradiance_matrix.tocsr()

    def determine_reachable_points_robot(self,sampling_places,world,robot,lamp,collider,show_vis = False,neighborhood = 0.4,float_height = 0.08,base_linknum = 2):
            
        self.set_robot_link_collision_margins(robot,0.03,collider)
        print(robot.numLinks())
        show_vis = True
        if(show_vis):
            vis.add('world',world)
            # eliminating draw distance
            vis.lock()
            # time.sleep(0.5)
            vp = vis.getViewport()
            # vp.h = 640
            # vp.w = 640
            vp.clippingplanes = [0.1,10000]
            tform = pickle.load(open('tform.p','rb'))
            vp.setTransform(tform)
            scale = 1
            vp.w = 1853//scale
            vp.h = 1123//scale
            # vis.setViewport(vp)
            vis.scene().setViewport(vp)
            vis.unlock()
            vis.show()
        
        reachable = []
        configs = []
        # world.terrain(0).geometry().setCollisionMargin(0.05)

        for place in tqdm(sampling_places):
            # pdb.set_trace()
            # time.sleep(0.2)
            reachable.append(self.solve_ik_near_sample(robot,lamp,collider,world,place,restarts = 10,tol = 1e-2,neighborhood = neighborhood,float_height = float_height))
            # print('reachable? = {}'.format(reachable[-1]))
            cfig = robot.getConfig()
        #         print(cfig[2])
            configs.append(cfig+place.tolist())
        
        # after checking for those margins, we reset the robot to its original configs for general motion planning
        # self.set_robot_link_collision_margins(robot,0,collider,self.angular_dofs)
        self.set_robot_link_collision_margins(robot,0,collider)
        # we also reset the base and environment collision to simplify path planning:
        world.terrain(0).geometry().setCollisionMargin(0)
        # base = robot.link(self.base_height_link)
        # base.geometry().setCollisionMargin(0)

    
        return reachable,configs
    
problem = TowerbotDisinfectionProblem(
    total_dofs = 4,
    linear_dofs = [0,1],
    angular_dofs = [],
    frozen_dofs = [2,3],
    base_height_link = 2,
    robot_height = 1.5,
    lamp_linknum = 2,
    lamp_local_coords = [0,0,0],
    active_dofs = [0,1,3,6,7,8,9],
    robot_cspace_generator = Robot3DCSpace,
    robot_cspace_solver = CSpaceObstacleSolver,
    float_height = 0.08)


experiment = '30_min'
tmax = 0.5
total_distances = []
coverages = []
resolutions = []

tmp = '/home/motion/Optimized-UV-Disinfection/data/environment_meshes/Single Scene Example - ScanNet0000_00/estimate_native_5_cm.ply'
for res in tqdm([750]):#,20,30,40,50,60,70,80]):
    total_distance,coverage,resolution,res_dir = problem.perform_experiment(
    results_dir = './3D_results',
    mesh_file = tmp,
    min_distance = 0.05,
    from_scratch = True,
    irradiance_from_scratch = True,
    float_height = 0.08,
    power = 80,
    resolution = res,
    experiment = experiment,
    tmax = tmax,
    robot_name = 'towerbot')
    total_distances.append(total_distance)
    coverages.append(coverage)
    resolutions.append(resolution)
    df = pd.DataFrame({'resolution':resolutions,'coverage':coverages,'total_distance':total_distances})
    df.to_csv(res_dir + '/3D_towerbot_results.csv', sep = '|', index = False)

