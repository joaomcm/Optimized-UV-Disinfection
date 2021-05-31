import pandas as pd
from tqdm import tqdm
from klampt.model.collide import WorldCollider
from klampt import WorldModel,Geometry3D
from klampt.plan.cspace import CSpace,MotionPlan


from planning.disinfection3d import DisinfectionProblem
from planning.robot_cspaces import Robot3DCSpace,CSpaceObstacleSolver 

import pdb
from klampt import vis
import time
import numpy as np
import pickle


class FloatbotDisinfectionProblem(DisinfectionProblem):
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
    float_height
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
        float_height)
    def setup_robot_and_light(self,robotfile = './data/idealbot.rob',
                                    mesh_file = './data/environment_meshes/full_detail_hospital_cad_meters.obj',float_height = 0.08):
        world = WorldModel()
        res1 = world.loadElement(robotfile)
        robot = world.robot(0)
        qmin,qmax = robot.getJointLimits()
        qmin = np.nan_to_num(qmin,posinf = 1000,neginf =-100)
        qmax = np.nan_to_num(qmax,posinf = 1000,neginf =-100)


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
        # collider.ignoreCollision((robot.link(8),robot.link(6)))
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

    def set_robot_link_collision_margins(self,robot,margin,collider,angular_dofs):
        
        for link_num in [2]:
            this_link = robot.link(link_num)
            this_link.geometry().setCollisionMargin(margin)


    def determine_reachable_points_robot(self,sampling_places,world,robot,lamp,collider,show_vis = False,neighborhood = 0.4,float_height = 0.08,base_linknum = 2):
            
        self.set_robot_link_collision_margins(robot,0.15,collider,[])

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

            robot.setConfig(place)
            
            reachable.append(self.solve_ik_near_sample(robot,lamp,collider,world,place,restarts = 10,tol = 1e-2,neighborhood = neighborhood,float_height = float_height))
            # print('reachable? = {}'.format(reachable[-1]))
            cfig = robot.getConfig()
        #         print(cfig[2])
            configs.append(cfig+place.tolist())
        
        # after checking for those margins, we reset the robot to its original configs for general motion planning
        # self.set_robot_link_collision_margins(robot,0,collider,self.angular_dofs)
        self.set_robot_link_collision_margins(robot,0,collider,[])
        # we also reset the base and environment collision to simplify path planning:
        world.terrain(0).geometry().setCollisionMargin(0)


    
        return reachable,configs

    def solve_ik_near_sample(self,robot,lamp,collider,world,place,restarts = 10,tol = 1e-3,neighborhood = 0.4,float_height = 0.08):
        robot.setConfig(place)
        if(self.collisionChecker(collider)):
            return True
        else:
            return False
    

class Floatbot3DCSpace(Robot3DCSpace):
        def __init__(self,bounds,robot,collider,lamp,milestones,base_height_link = 2,robot_height = 1.5,float_height = 0.08,linear_dofs = [0,1],angular_dofs = [4,5,6,7,8],light_local_position  = [0,0,0]):
            CSpace.__init__(self)

            self.base_height_link = base_height_link
            #set bounds
            limits = robot.getJointLimits()
            limits[0][0] = 0
            limits[0][1] = 0
            limits[0][self.base_height_link] = 0
            limits[1][0] = bounds[0]
            limits[1][1] = bounds[1]
            limits[1][self.base_height_link] = 2
            # we also set all the joint limits of all joints that are not active to be 
            # equal to their current positions:
            cfig = robot.getConfig()
            self.zero_cfig = np.array(cfig)
            tmp = set(linear_dofs+angular_dofs)
            tmp2 = set(list(range(len(cfig))))
            fixed_dofs = list(tmp2.difference(tmp))
            self.fixed_dofs = fixed_dofs
            for fixed_dof in fixed_dofs:
                limits[0][fixed_dof] = cfig[fixed_dof]
                limits[1][fixed_dof] = cfig[fixed_dof]
            robot.setJointLimits(limits[0],limits[1])
            self.lamp = lamp
            self.max_vals = limits[1] + [bounds[0],bounds[1],robot_height]
            self.min_vals = limits[0] + [0,0,0]
            bound = []
            for a,b in zip(self.min_vals,self.max_vals):
                bound.append((a,b))
            self.bound = bound
            self.milestones = milestones
            #set collision checking resolution
            self.eps = 0.01
            #setup a robot with radius 0.05
            self.robot = robot
            #set obstacles here
            self.collider = collider
            self.remaining_milestones = self.milestones[:]
            self.fraction = 1
            self.remaining = set(range(self.remaining_milestones.shape[0]))
            self.S = None
            self.clf = None
            self.linear_dofs = linear_dofs
            self.angular_dofs = angular_dofs
            self.local_lamp_coords = light_local_position
            self.counter = 0


problem = FloatbotDisinfectionProblem(
    total_dofs = 3,
    linear_dofs = [0,1,2],
    angular_dofs = [],
    frozen_dofs = [],
    base_height_link = 2,
    robot_height = 1.5,
    lamp_linknum = 2,
    lamp_local_coords = [0,0,0],
    active_dofs = [0,1,2],
    robot_cspace_generator = Floatbot3DCSpace,
    robot_cspace_solver = CSpaceObstacleSolver,
    float_height = 0.08)



experiment = '30_min'
tmax = 0.5
total_distances = []
coverages = []
resolutions = []

tmp = './data/environment_meshes/remeshed_hospital_room_full_35k.obj'
for res in tqdm([500]):#,20,30,40,50,60,70,80]):
    total_distance,coverage,resolution,res_dir = problem.perform_experiment(
    results_dir = './3D_results',
    mesh_file = './data/environment_meshes/remeshed_hospital_room_full_35k.obj',
    min_distance = 0.05,
    from_scratch = True,
    irradiance_from_scratch = True,
    float_height = 0.08,
    power = 80,
    resolution = res,
    experiment = experiment,
    tmax = tmax,
    robot_name = 'floatbot')
    total_distances.append(total_distance)
    coverages.append(coverage)
    resolutions.append(resolution)
    df = pd.DataFrame({'resolution':resolutions,'coverage':coverages,'total_distance':total_distances})
    df.to_csv(res_dir + '/3D_floabot_results.csv', sep = '|', index = False)

