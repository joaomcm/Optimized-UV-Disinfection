"""
    Copyright 2021 University of Illinois Board of Trustees. All Rights Reserved.

    Licensed under the “Non-exclusive Research Use License” (the "License");

    The License is included in the distribution as LICENSE.txt file.

    See the License for the specific language governing permissions and imitations under the License.

"""



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
from visibility_new.visibility import Visibility
from joblib import Parallel,delayed
from tqdm import tqdm
from scipy.sparse import lil_matrix

from gurobipy import GRB
from joblib import Parallel,delayed

from sklearn.neighbors import KNeighborsClassifier

from getToursAndPaths import getTour, readTourFile, getPathFromPrm, getFinalPath


class Robot_3D_CSpace(CSpace):
    def __init__(self,bounds,robot,collider,lamp,milestones,base_height_link = 2,robot_height = 1.5,float_height = 0.08,linear_dofs = [0,1],angular_dofs = [4,5,6,7,8],light_local_position  = [0,0,0]):
        CSpace.__init__(self)
        self.base_height_link = base_height_link
        #set bounds
        limits = robot.getJointLimits()
        limits[0][0] = 0
        limits[0][1] = 0
        limits[0][self.base_height_link] = float_height
        limits[1][0] = bounds[0]
        limits[1][1] = bounds[1]
        limits[1][self.base_height_link] = float_height
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

    def collisionChecker(self):
        if(list(self.collider.collisions())!= []):
            return False

        return True
    
    def create_internal_subgraphs(self,G_list):
        G = nx.Graph()
        self.G_list = G_list
        G.add_nodes_from(range(len(G_list[0])))
        G.add_edges_from(G_list[1])

        self.S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        main_nodes = np.array(list(self.S[0]))
        features = np.array(G_list[0])[main_nodes,self.robot.numLinks():]
        self.clf = KNeighborsClassifier(1,n_jobs = -4)
        self.clf.fit(features,main_nodes)

    def select_disjoint_to_sample(self):


        # we then sample one node from the remaining nodes:
        focus_node = np.random.choice(list(self.remaining))
        #find its connected component
        # we then find the subgraph that contains one of the non-connected nodes:
        for this_s in self.S[1:]:
            if(this_s.has_node(focus_node)):
                break
        focus_nodes = np.array(list(this_s.nodes))
        focus_features =  np.array(self.G_list[0])[focus_nodes,self.robot.numLinks():]
        distances,focus_neighbor = self.clf.kneighbors(focus_features)
        #         print(focus_neighbor)
        nearest = np.argmin(distances)
        nearest_component = focus_nodes[nearest]
        distances,nearest_connected = self.clf.kneighbors(np.array(self.G_list[0])[focus_node,self.robot.numLinks():].reshape(1,-1))
        nearest_connected = nearest_connected[0][0]
        #         print(nearest_connected)
        return self.G_list[0][nearest_component],self.G_list[0][nearest_connected]
    
    def interp(self,m_a,m_b,steps = 25):
        divs = np.linspace(0,1,num = steps)
        dif = m_b-m_a
        interm_steps = m_a + np.matmul(divs.reshape(-1,1),dif.reshape(1,-1))
        return interm_steps
    
    def set_remaining_milestones(self,remaining,G_list):
        self.remaining = remaining
        self.remaining_milestones = self.milestones[list(remaining)]
        self.fraction = len(remaining)/self.milestones.shape[0]
        self.create_internal_subgraphs(G_list)
        # if(self.fraction < 0.2):
        #     vis.add('remaining_milestones',[i[:12] for i in self.remaining_milestones],color = [1,0,0,0.5])
        
    def feasible(self,q):

        self.robot.setConfig(q[:self.robot.numLinks()])
        
        
        if(self.collisionChecker()):
            return True
        else:
            return False
        
    def visible(self,a,b):
        #         print('checking visibility')
        q_a = np.array(a[:self.robot.numLinks()])
        q_b = np.array(b[:self.robot.numLinks()])
        interm = self.interp(q_a,q_b)
        for cfig in interm:
            self.robot.setConfig(cfig)
            # if we are in collision, points are not visible
            if(not self.collisionChecker()):
                return False
        # if no collisons are detected - then we are visible!
        return True
            
    def distance(self,a,b):
        x_a = np.array(a[self.robot.numLinks():])
        x_b = np.array(b[self.robot.numLinks():])
        dist = float(np.linalg.norm(x_a-x_b))
        if(dist <= 1):
            return self.compute_actual_distance(np.array(a[:self.robot.numLinks()]),np.array(b[:self.robot.numLinks()]))
        else:
            return dist
        
    def sample_nearby(self,milestone,angle_neighborhood = np.pi/12,spatial_neighborhood = 0.1,debug = False):
        tries = 100
        this_min = self.min_vals.copy()
        this_max = self.max_vals.copy()

        for i in range(self.linear_dofs):
            this_min[i] = milestone[i]-spatial_neighborhood
            # this_min[1] = milestone[1]-spatial_neighborhood
            this_max[i] = milestone[i]+spatial_neighborhood
            # this_max[1] = milestone[1]+spatial_neighborhood
        # we then do the same for all the angles:
        for i in self.angular_dofs:
            this_min[i] = max(milestone[i]-angle_neighborhood,self.min_vals[i])
            this_max[i] = min(milestone[i]+angle_neighborhood,self.max_vals[i])
        #             print(this_min[:2],this_max[:2])
        valid = False
        for i in range(tries):
            cfig = np.random.uniform(this_min[:self.robot.numLinks()],this_max[:self.robot.numLinks()])
            self.robot.setConfig(cfig)
            if(debug):
                print(cfig)
            if(self.collisionChecker()):
        #                     valid = True
                new_cfig = self.robot.getConfig()
                # new_cfig[9] = 0
                # we then get the cartesian position:
                xyz = self.lamp.getWorldPosition(self.local_lamp_coords)
                if(debug):
                    print('done sampling near milestone')
                return new_cfig + xyz
        return self.sample()
    
    def compute_actual_distance(self,origin,end):
        #         for origin,end in zip(origins,ends):
        interp = self.interp(origin,end)
        interp = interp[:,:self.robot.numLinks()]
        positions = []
        for cfig in interp:
            self.robot.setConfig(cfig)
            positions.append(self.lamp.getWorldPosition(self.local_lamp_coords))
        positions = np.array(positions)
        distance = np.linalg.norm(np.diff(positions,axis = 0),axis = 1).sum()
        return distance
    
    def sample(self):
        explore = 0.3
        neighborhood = 1
        tries = 100
        cfig = self.robot.getConfig()
        debug = False
        #         print(self.fraction)
        if(self.fraction > 0.2):
            if(np.random.rand() < explore):
                valid = False
                if(debug):
                    print('started sampling random')

                while(not valid):
                    # last dof is irrelevant since the light is symmetrical 
                    self.robot.randomizeConfig()
                    if(self.collisionChecker()):
                        valid = True
                        new_cfig = self.robot.getConfig()
                        new_cfig[9] = 0
                        # we then get the cartesian position:
                        xyz = self.lamp.getWorldPosition(self.local_lamp_coords)
                        if(debug):
                            print('done sampling random')
                        return new_cfig + xyz
            # otherwise, we sample close to one of the remaining milestones
            else:
                if(debug):
                    print('sampling near milestone')
                # we first randomly select from the pool of remaining milestones to be connected:
                chosen_index = np.random.choice(range(len(self.remaining_milestones)))
                milestone = self.remaining_milestones[chosen_index].copy()
                  #             print(milestone)
                this_min = self.min_vals.copy()
                this_max = self.max_vals.copy()

                for i in self.linear_dofs:
                    this_min[i] = milestone[i]-neighborhood
                    # this_min[1] = milestone[1]-neighborhood
                    this_max[i] = milestone[i]+neighborhood
                    # this_max[1] = milestone[1]+neighborhood
                #             print(this_min[:2],this_max[:2])
                valid = False
                for i in range(tries):
                    cfig = np.random.uniform(this_min[:self.robot.numLinks()],this_max[:self.robot.numLinks()])
                    # print(len(cfig))
                    self.robot.setConfig(cfig)
                    if(debug):
                        print(cfig)
                    if(self.collisionChecker()):
                    #                     valid = True
                        new_cfig = self.robot.getConfig()
                        new_cfig[9] = 0
                        # we then get the cartesian position:
                        xyz = self.lamp.getWorldPosition(self.local_lamp_coords)
                        if(debug):
                            print('done sampling near milestone')
                        return new_cfig + xyz
                return self.sample()
        #otherwise, perform agressive sampling near the not connected components
        else:
            #             print('sampling nearby!! Focused strategy!')
            focus,nearest_connected = self.select_disjoint_to_sample()
            # we then randomly sample near the focus point or the nearest connected component to it:
            if(np.random.rand() > 0.7):
                return self.sample_nearby(focus)   
            else:
                return self.sample_nearby(focus,spatial_neighborhood = 0.3,angle_neighborhood = np.pi/6)


class CSpaceObstacleSolver:
        def __init__(self,space,milestones = (0,0),initial_points = 4000,steps = 100):
            self.space = space
            #PRM planner
            MotionPlan.setOptions(type="prm",knn=10,connectionThreshold=1,ignoreConnectedComponents = True)
            self.optimizingPlanner = False
            # we first plan a bit without goals just to have a good skeleton
            self.initial_points = initial_points
            self.steps = steps
            self.milestones = milestones
            self.times = 0
            self.planner = MotionPlan(space)
            for milestone in self.milestones:
                self.planner.addMilestone(milestone.tolist())
            self.components = int(self.planner.getStats()['numComponents'])
            print(self.components)
            print('planning initial roadmap with {} points'.format(initial_points))

            self.planner.planMore(self.initial_points)
            self.connected = False
            # we now add each of the chosen points as a milestone:
            self.G = self.planner.getRoadmap()
            self.start_milestones = len(self.G[0])
            
            self.path = []
            self.milestone_2 = 1
            self.G = None
            self.count = 0
            self.connected_list = {0}
            self.total_milestones = set(list(range(len(self.milestones))))
            
        def compute_actual_distance(self,origins,ends):
                weights = []
                for origin,end in zip(origins,ends):
                    interp = self.space.interp(origin,end)
                    interp = interp[:,:self.space.robot.numLinks()]
                    positions = []
                    for cfig in interp:
                        self.space.robot.setConfig(cfig)
                        positions.append(self.space.lamp.getTransform()[1])
                    positions = np.array(positions)
                    distance = np.linalg.norm(np.diff(positions,axis = 0),axis = 1).sum()
                    weights.append(distance)
                return weights
            
        def compute_real_pairwise_distances(self,G_list):
            G = nx.Graph()
            G.add_nodes_from(range(len(G_list[0])))
            G.add_edges_from(G_list[1])
            edges = np.array(G_list[1])
            nodes = np.array(G_list[0])
            origins = nodes[edges[:,0],:self.space.robot.numLinks()]
            ends = nodes[edges[:,1],:self.space.robot.numLinks()]
            weights = self.compute_actual_distance(origins,ends)
            for weight,edge in zip(weights,edges):
                G.edges[edge]['weight'] = weight
            distances_array = []
            for i in tqdm(range(self.milestones.shape[0])):
                distances_dict = dict(nx.algorithms.shortest_path_length(G,source = i, weight = 'weight'))
                this_distance = []
                for j in range(self.milestones.shape[0]):
                    this_distance.append(distances_dict[j])
                distances_array.append(this_distance)
            distances = np.array(distances_array)
            self.G = G
            return distances


        def get_adjacency_matrix_from_milestones(self):
            
            while(self.connected == False):
                if(self.space.fraction > 0.2):
                    print( "Planning {}... components = {}".format(self.steps,int(self.planner.getStats()['numComponents'])))
                else:
                    print( "Focused Planning {}... components = {}".format(self.steps,int(self.planner.getStats()['numComponents'])))
                self.planner.planMore(self.steps)
                
                milestone_1 = 0
                remaining = self.total_milestones - self.connected_list
                G_list = self.planner.getRoadmap()
                G = nx.Graph()
                self.space.set_remaining_milestones(remaining,G_list)
                G.add_nodes_from(range(len(G_list[0])))
                G.add_edges_from(G_list[1])
                elements_with_zero = nx.algorithms.node_connected_component(G,0)
                self.connected_list = self.total_milestones.intersection(elements_with_zero)
                print('connected so far: ',len(self.connected_list))

                if(self.connected_list == self.total_milestones):
                    self.connected = True
                if(self.space.fraction < 0.2):
                    self.steps = 20
            print('PRM connecting all milestones found - computing adjacency matrix')
            rm = self.planner.getRoadmap()
            self.adjacency_matrix = self.compute_real_pairwise_distances(rm)
            print('calculated all distances')

                
            return self.adjacency_matrix,self.G,rm[0]