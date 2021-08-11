"""
    Copyright 2021 University of Illinois Board of Trustees. All Rights Reserved.

    Licensed under the “Non-exclusive Research Use License” (the "License");

    The License is included in the distribution as LICENSE.txt file.

    See the License for the specific language governing permissions and imitations under the License.

"""



import numpy as np
from shapely.geometry import Polygon, LineString, LinearRing, MultiPolygon,Point,MultiPoint 
from shapely.geometry.collection import GeometryCollection
import math
from klampt.plan.cspace import CSpace,MotionPlan
from klampt.vis.glprogram import GLProgram
from klampt.math import vectorops
from tqdm import tqdm 
import networkx as nx


def interpolate(a,b,u):
    """Interpolates linearly between a and b"""
    return vectorops.madd(a,vectorops.sub(b,a),u)

class Circle:
    def __init__(self,x=0,y=0,radius=1):
        self.center = (x,y)
        self.radius = radius
        
    def contains(self,point):
        return (vectorops.distance(point,self.center) <= self.radius)

class MultiPgon:
    def __init__(self,poly,y_bound):
        self.poly = poly
        self.y_bound = y_bound
    def contains(self,point):
        return self.poly.distance(Point(point[0],point[1])) < 0.05
    def distance(self,a,b):
        return self.poly.distance(LineString([(a[0],a[1]),(b[0],b[1])]))
    
class CircleObstacleCSpace(CSpace):
    def __init__(self,x_bound =3.0,y_bound=3.0):
        CSpace.__init__(self)
        #set bounds
        self.bound = [(-2.0,x_bound),(-2.0,y_bound)]
        #set collision checking resolution
        self.eps = 0.025
        #setup a robot with radius 0.05
        self.robot = Circle(0,0,0.05)
        #set obstacles here
        self.obstacles = []
        self.fraction = 1

    def addObstacle(self,circle):
        self.obstacles.append(circle)
    
    def feasible(self,q):
        #bounds test
#         if not CSpace.feasible(self,q): return False
        #TODO: Problem 1: implement your feasibility tests here
        #currently, only the center is checked, so the robot collides
        #with boundary and obstacles
        for o in self.obstacles:
            if o.contains(q): return False
        return True

#     def visible(self,a,b):
#         # print('\n\n\n')
#         # print(a,b)
#         # print('\n\n\n')

#         for o in self.obstacles:
# #             print(o)
#             if(o.distance(a,b) < 0.1): 
#                 # print(o.distance(a,b))      
#                 # print('not visible!')
#                 return False
#         return True



class CSpaceObstacleSolver:
    def __init__(self,space,x_bound = 1.0,y_bound = 1.0,milestones = (0,0),initial_points = 4000):
        self.space = space
        #PRM planner
        MotionPlan.setOptions(type="prm",knn=10,connectionThreshold=0.05,ignoreConnectedComponents = True)
        self.optimizingPlanner = False
        # we first plan a bit without goals just to have a good skeleton
        self.initial_points = initial_points
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.milestones = milestones
        self.times = 0
        self.planner = MotionPlan(space)
        print('planning initial roadmap with {} points'.format(initial_points))
        self.planner.planMore(self.initial_points)
        self.connected = False
        # we now add each of the chosen points as a milestone:
        self.G = self.planner.getRoadmap()
        self.start_milestones = len(self.G[0])
        for milestone in self.milestones:
            self.planner.addMilestone(milestone)
        self.components = int(self.planner.getStats()['numComponents'])
        print(self.components)

        self.path = []
        self.G = None
    def get_adjacency_matrix_from_milestones(self):
        while(self.components > 1):
            print( "Planning 100...")
            self.planner.planMore(100)
            self.path = self.planner.getPath()
            self.G = self.planner.getRoadmap()
            self.components = int(self.planner.getStats()['numComponents'])
            print(self.components)
            
        print('PRM connecting all milestones found - computing adjacency matrix')
        pathDict = dict()
        
        self.adjacency_matrix = np.zeros(shape = (len(self.milestones),len(self.milestones)))
        self.adjacency_matrix[:,:] = np.inf
        for i,milestone1 in tqdm(enumerate(range(self.start_milestones,self.start_milestones+1 + len(self.milestones)))):
#             print(self.G[0][milestone1])
            for j,milestone2 in enumerate(range(milestone1+1,self.start_milestones + len(self.milestones))):
                j = j+i + 1
                path = self.planner.getPath(milestone1,milestone2)
                cost = self.planner.pathCost(path)
                self.adjacency_matrix[i,j] = cost
                self.adjacency_matrix[j,i] = cost
                pathDict[i,j] = pathDict[j,i] = path

        #         print((i,j),(j,i))
        print('calculated all distances')
        for i in range(self.adjacency_matrix.shape[0]):
            self.adjacency_matrix[i,i] = 0
            
        return self.adjacency_matrix, pathDict



# creating the space

# space = robotplanning.makeSpace(world,robot,edgeCheckResolution=0.05,movingSubset = [0,1,4,5,6,7,8])

settings = {'type':'rrt',
    'perturbationRadius':0.25,
    'bidirectional':True,
    'shortcut':False,
    'restart':False,
    'restartTermCond':"{foundSolution:1,maxIters:3000}"}


class Robot_2D_CSpace(CSpace):
    def __init__(self,bounds,milestones):
        CSpace.__init__(self)
        #set bounds
        bound = []
        for a,b in zip(self.min_vals[:2],self.max_vals[:2]):
            bound.append((a,b))
        self.bound = bound
        self.milestones = milestones
        #set collision checking resolution
        self.eps = 0.05
        #setup a robot with radius 0.05
        #set obstacles here
        self.remaining_milestones = self.milestones[:]
        self.fraction = 1
#         self.remaining = set(range(self.remaining_milestones.shape[0]))
#         self.S = None
#         self.clf = None
    
    def addObstacle(self,circle):
        self.obstacles.append(circle)
    
    def feasible(self,q):
        #bounds test
#         if not CSpace.feasible(self,q): return False
        #TODO: Problem 1: implement your feasibility tests here
        #currently, only the center is checked, so the robot collides
        #with boundary and obstacles
        for o in self.obstacles:
            if o.contains(q): return False
        return True
                
                


class CSpaceObstacleSolver1:
    def __init__(self,space,milestones = (0,0),initial_points = 4000,steps = 100):
        self.space = space
        #PRM planner
        MotionPlan.setOptions(type="prm",knn=10, connectionThreshold=1,ignoreConnectedComponents = True)
        self.optimizingPlanner = False
        # we first plan a bit without goals just to have a good skeleton
        self.initial_points = initial_points
        self.steps = steps
        self.milestones = milestones
        self.times = 0
        self.planner = MotionPlan(space)
        for milestone in self.milestones:
            self.planner.addMilestone(milestone)
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
#             weights = []
            weights = np.linalg.norm(ends-origins,axis = 1)
        
            return weights
        
    def compute_real_pairwise_distances(self,G_list):
        G = nx.Graph()
        G.add_nodes_from(range(len(G_list[0])))
        G.add_edges_from(G_list[1])
        edges = np.array(G_list[1])
        nodes = np.array(G_list[0])
#         for config,node in zip(G_list[0],nodes):
#             G.nodes[node]['config'] = config
        origins = nodes[edges[:,0]]
        ends = nodes[edges[:,1]]
        weights = self.compute_actual_distance(origins,ends)
        for weight,edge in zip(weights,edges):
            G.edges[edge]['weight'] = weight
        print('actually calculating distances')
        self.G = G
        distances_array = []
        for i in tqdm(range(len(self.milestones))):
            distances_dict = dict(nx.algorithms.shortest_path_length(G,source = i, weight = 'weight'))
            this_distance = []
            for j in range(len(self.milestones)):
                this_distance.append(distances_dict[j])
            distances_array.append(this_distance)
        distances = np.array(distances_array)
#         self.G = G
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
#             self.space.set_remaining_milestones(remaining,G_list)
            G.add_nodes_from(range(len(G_list[0])))
            G.add_edges_from(G_list[1])
            elements_with_zero = nx.algorithms.node_connected_component(G,0)
            self.connected_list = self.total_milestones.intersection(elements_with_zero)
            print('connected so far: ',len(self.connected_list))
#             self.components = 
#             for milestone in remaining:
# #                 print(milestone_1,milestone)
#                 if(not(self.planner.getPath(milestone_1,milestone) == None)):
#                     self.connected_list.update([milestone])
            if(self.connected_list == self.total_milestones):
                self.connected = True
            if(self.space.fraction < 0.2):
                self.steps = 20
        print('PRM connecting all milestones found - computing adjacency matrix')
        pathDict = dict()
        
        self.adjacency_matrix = np.zeros(shape = (len(self.milestones),len(self.milestones)))
        # self.adjacency_matrix[:,:] = 0

#         for i,milestone1 in tqdm(enumerate(range(0,1 + len(self.milestones)))):
# #             print(self.G[0][milestone1])
#             for j,milestone2 in enumerate(range(milestone1+1,len(self.milestones))):
#                 j = j+i + 1
#                 path = self.planner.getPath(milestone1,milestone2)
#                 cost = self.planner.pathCost(path)
#                 self.adjacency_matrix[i,j] = cost
#                 self.adjacency_matrix[j,i] = cost
#                 pathDict[i,j] = pathDict[j,i] = path
#                 print((i,j),(j,i))
        rm = self.planner.getRoadmap()
        self.adjacency_matrix = self.compute_real_pairwise_distances(rm)
        print('calculated all distances')

            
        return self.adjacency_matrix,rm