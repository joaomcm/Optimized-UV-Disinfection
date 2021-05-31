
# coding: utf-8

# In[ ]:

max_skew = 30
max_x = 50
max_y = 50
max_scale = 5
max_rot = 120
from shapely import speedups
if (speedups.available):
    speedups.enable()
    print('enabling speedups')
import numpy as np
from shapely.geometry import Polygon, LineString, LinearRing, MultiPolygon,Point
import math
from shapely.affinity import rotate,scale,skew,translate
from shapely.ops import substring
import pickle
import os
from matplotlib import patches 
from matplotlib import pyplot as plt
from figures import BLUE, SIZE, set_limits, plot_coords, color_isvalid
from shapely.geometry import MultiPoint,Point
from shapely.ops import split
from tqdm import tqdm
import shapely
from importlib import reload
import gridLP
import visibilityGraph as vs
reload(gridLP)
from gridLP import get_grid_points, get_vs_graphs, solve_lp, get_grid_points_res
from shapely.geometry.collection import GeometryCollection
import plotSolutions
reload(plotSolutions)
import plotSolutions as myplot

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import math
from klampt.plan.cspace import CSpace,MotionPlan
from klampt.vis.glprogram import GLProgram
from klampt.math import vectorops
class Scene_Creator:

    def __init__(self,max_skew = 30,max_x = 50, max_y = 50, max_scale = 5, max_rot = 120,min_clearance = 0.1,resolution = 1000):
        """
        Initializes the instance of the Scene Creator with its parameters.

        Args:
            max_skew (float): maximum angle by which each individual shape might be skewed, default = 30
            max_x (float): Maximum X coordinate of the bounding box, default = 50
            max_y (float): Maximum Y coordinate of the bounding box, default = 50
            max_scale (float): Maximum scaling of each individual polygon , default = 5 (i.e. 5 is the largest side of a square)
            max_rot (float): maximum angle of rotation of each individual polygon
        Returns:
             
        """
        k = np.sqrt(3)
        self.max_skew = max_skew
        self.max_x = max_x
        self.max_y = max_y
        self.max_scale = max_scale
        self.max_rot = max_rot
        self.min_area_threshold = 0.0005
        self.min_clearance = min_clearance
        square = [(0.0,0.0),(1.0,0.0),(1.0,1.0),(0.0,1.0)]
        triangle = [(0.0,0.0),(1.0,0.0),(0.5,k/3)]
        hexagon = [(0.0,k/2.0),(1.0/2,0.0),(3.0/2.0,0.0),(2,k/2.0),(3.0/2.0,k),(1.0/2.0,k)]
        self.primitives = {'square':square,'triangle':triangle,'hexagon':hexagon}
        self.resolution = resolution
    def create_random_shape(self):
        """
        Creates a polygon whose shape is selected from 3 primitives 
        (a unit cube, a unit equilateral triangle and a unit hexagon).That shape is 
        then randomly transformed, by randomly skewing it by a random angle in [0,max_skew],
        rotating it by a random angle in [0,max_rot], 
        scaling it by a random scale in [0,max_scale] and translating it by a random vector [x,y],
        where x is in [0,max_x] and y is in [0,max_y]

        Args: None

        Returns: pol (Polygon instance from shapely) created as described above
             
        """
        chosen = np.random.choice(list(self.primitives.keys()))

        pol = Polygon(self.primitives[chosen])
        rand_trans_x = self.max_x*np.random.random()
        rand_trans_y = self.max_y*np.random.random()
        pol = translate(pol,rand_trans_x,rand_trans_y)
        rand_scale = self.max_scale*np.random.random(2)
        pol = scale(pol,rand_scale[0],rand_scale[1])
        rot = self.max_rot*np.random.random()
        pol = rotate(pol,rot)
        rand_skew = self.max_skew-2.0*self.max_skew*np.random.random(2)
        pol = skew(pol,rand_skew[0],rand_skew[1])



        return pol
    def create_scene(self,total_polygons = 30,max_skew = None,max_x = None, max_y = None, max_scale = None, max_rot = None,max_tries = 5,min_clearance = 0.1,resolution = None):
        """
        Creates a Scene instance consisting of a MultiPloygon instance with total_polygons number of polygons
        which are created according to the method described in create_random_shape above.
        The scene generated has to be valid (i.e. intersection between polygons has to have zero area). If after sampling
        total_polygons*max_tries polygons it is unable to find a scene with that many polygons, it will return the maximum number
        of polygons it was able to fit in the scene, alongside an error message warning the user that the total number
        of polygons contained in the scene is smaller than the commanded. 

        Args:
            total_polygons (int): total number of polygons that you wish to include in the scene
            max_skew (float): maximum angle by which each individual shape might be skewed, default = 30
            max_x (float): Maximum X coordinate of the bounding box, default = 50
            max_y (float): Maximum Y coordinate of the bounding box, default = 50
            max_scale (float): Maximum scaling of each individual polygon , default = 5 (i.e. 5 is the largest side of a square)
            max_rot (float): Maximum angle of rotation of each individual polygon
            max_tries (int): Number of attempts this code is allowed to sample from random polygons
            in its attempt to create a valid scene with that number of polygons. The actual number of attempts is 
            given by max_tries*total_polygons.
            min_clearance(float): minimum distance between all the geometries in the scene
        Returns:
            scene (Scene) a Scene instance containing the selected number of randomly sampled polygons
        """
        if(max_skew):
            self.max_skew = max_skew
        if(max_x):
            self.max_x = max_x
        if(max_y):
            self.max_y = max_y
        if(max_scale):
            self.max_scale = max_scale
        if(max_rot):
            self.max_rot = max_rot
        if(min_clearance):
            self.min_clearance = min_clearance
        if(resolution):
            self.resolution = resolution
        max_tries = max_tries*total_polygons
        included_polygons = []
        acceptable_polygons = 0
        tries = 0
        poly = self.create_random_shape()
        poly = self.cut_polygon_to_resolution(poly,self.resolution)
        included_polygons = [poly]
        mp = MultiPolygon(included_polygons)
        while(acceptable_polygons < total_polygons):
            shape = self.create_random_shape()
#             print(shape.area)
#             display(shape)
            if(shape.area > self.min_area_threshold):
                distance = shape.distance(mp)
                if(distance > self.min_clearance):
#                     print(mp.is_valid)
                    shape = self.cut_polygon_to_resolution(shape,self.resolution)
                    included_polygons += [shape]
#                     print(included_polygons)
                    mp = MultiPolygon(included_polygons)
                    acceptable_polygons += 1
            tries += 1
            if(tries > max_tries):
                print("""Failed to Produce a viable solution. Increase the bounding box 
                      (max_x,max_y),reduce the number of polygons in the scene or 
                      try again with the same parameters but higher max_tries multiplier
                      Default (5). 
                      Returning the scene with maximum number of polygons within the 
                      maximum number of tries""")
                return Scene(MultiPolygon(included_polygons))
            
        print("Scene Generated Sucessfully. Returning the scene")
        return Scene(MultiPolygon(included_polygons))
    def load_scene(self,file_path):
        """
        loads a scene that was previously saved as a pickle file by a scene instance.
        Args: filename (str) - a string indicating the filepath to the scene
        Returns: 
             
        """
        return pickle.load(open(file_path,'rb'))
    def create_scene_from_specs(self,specs):
        polys = []
        for poly in specs:
            polys.append(Polygon(poly))
        return Scene(MultiPolygon(polys))
    def cut_polygon_to_resolution(self,poly,resolution):
        """cuts the straight lines into equally sized line segments no larger than the prescribed resolution"""
        segments = []
        poly_coords = list(poly.exterior.coords)
        for i in range(len(poly_coords)-1):
            line = LineString((poly_coords[i],poly_coords[i+1]))
            # we here determine the actual resolution of the polygons
            divs = line.length/(np.ceil(line.length/resolution))
            
            for j in np.arange(0,line.length,divs):
                segment = substring(line,j,j+divs)
                segments.extend(list(segment.coords)[1:2])
        final_polygon = Polygon(segments[:])
        return final_polygon
    def remake_scene_to_resolution(self,scene,resolution):
        final_polygons = []
        for poly in scene.scene.geoms:
            poly = self.cut_polygon_to_resolution(poly,resolution)
            final_polygons.append(poly)
        return Scene(MultiPolygon(final_polygons))
            
class Scene:
    def __init__(self,scene):
        """
        Initializes the Scene Instance.

        Args:
            scene (MultiPolygon) : A MultiPolygon instance that describes the obstacles in a scene
        Returns:
             
        """
        self.scene = scene
    def display_ipython(self):
        """
        Displays the scene in ipython consoles and Jupyter notebook.

        Args:
        Returns:
             
        """
        display(self.scene)
    def display_matplotlib(self):
        fig = plt.figure(1,figsize= SIZE,dpi = 90)
        ax = fig.add_subplot(111)
        for polygon in self.scene:
            plot_coords(ax,polygon.exterior)
            patch = patches.Polygon(polygon.exterior.xy, facecolor=color_isvalid(self.scene), edgecolor=color_isvalid(self.scene, valid=BLUE), alpha=0.5, zorder=2)
            ax.add_patch(patch)
        plt.show()
    def get_polygon_coordinates(self):
        """
        Provides the coordinates of the polygons that form the scene as a list, with their
        vertices listed in counter-clockwise direction as a list of tuples
        Args:
        Returns: obstacle_coordinates (list), a list of lists, where each list contains the
        coordinates of each vertex of a given polygon in the scene as tuples, listed in
        counter-clockwise fashion.
             
        """
        obstacle_coordinates = []
        for poly in list(self.scene.geoms):
            obstacle_coordinates.append(list(poly.exterior.coords))
        return obstacle_coordinates
    def get_vertex_coordinates(self):
        """
        Provides the coordinates of the polygons that form the scene as a list, with their
        vertices listed in counter-clockwise direction as a list of tuples
        Args:
        Returns: obstacle_coordinates (list), a list of tuples, where each tuple is a vertex of a 
        in the scene
             
        """
        obstacle_coordinates = []
        for poly in list(self.scene.geoms):
            obstacle_coordinates.extend(list(poly.exterior.coords))
        return obstacle_coordinates
        
    def get_edges(self):
        """
        Provides the edges of the polygons that form the scene as a list, with their
        vertices listed in counter-clockwise direction as a list of tuples of 2 tuples
        Args:
        Returns: obstacle_coordinates (list), a list of tuples, where each list contains a tuple representing
        each of the edges in the polygons in the scene counter-clockwise fashion.
             
        """
        edges = []
        for poly in list(self.scene.geoms):
            obstacle_coordinates = list(poly.exterior.coords)
            for i in range(len(obstacle_coordinates)-1):
                edges.append((obstacle_coordinates[i],obstacle_coordinates[i+1]))
        return edges
    def get_bbox(self):
        return self.scene.bounds
        
    def save(self,filename):
        """
        Saves a scene as a pickle file under the name fileame to be loaded later via Scene_Creator.load()
        Args: filename (str) - a string of the path where you wish to save your scene.
        Returns: 
             
        """
        pickle.dump(self,open(filename,'wb'))
        


# In[ ]:

# try:
#     del sc
# except:
#     pass
# sc = Scene_Creator()
# scene = sc.create_scene(total_polygons = 20,max_x =2.0,max_y = 2.0,max_scale = 0.75,max_tries = 100,max_skew = 15,max_rot = 60, min_clearance = 0.1,resolution = 1000 )
# poly = scene.scene.geoms[0]

# scene.display_matplotlib()


# In[ ]:

# scene2 = sc.remake_scene_to_resolution(scene,0.1)
# scene2.display_matplotlib()


# In[ ]:

#### Things I've changed
#gridPointList = get_grid_points_res(x_res = 0.1,y_res = 0.1,xBounds = [0,bbox[2]],yBounds = [0,bbox[3]],minDist = 0.1,obstacles = scene.scene)
#bbox = scene2.get_bbox()

#### Things I've changed
#gridPointList = get_grid_points_res(x_res = 0.1,y_res = 0.1,xBounds = [0,bbox[2]],yBounds = [0,bbox[3]],minDist = 0.1,obstacles = scene.scene)
#gridPointList = get_grid_points_res(x_res = 0.1,y_res = 0.1,xBounds = [bbox[0]-0.5,bbox[2]+0.5],yBounds = [bbox[1]-0.5,bbox[3]+0.5],minDist = 0.1,obstacles = scene2.scene)


# In[ ]:

# a = sc.cut_polygon_to_resolution(poly,110)
# print(list(a.exterior.coords))

# s = a.exterior.coords
# print(len(list(s)),len(set(s)))


# In[ ]:

# initialize the scene creator
# sc = Scene_Creator()
# call create_scene to generate a Scene object

#### What I've changed
# scene = sc.create_scene(total_polygons = 15,max_x =2.0,max_y = 2.0,max_scale = 0.75,max_tries = 100,max_skew = 15,max_rot = 60, min_clearance = 0.2 )
# testList = [
# [(4.5, 0.5),(4.5, 4.5),(2.5, 4.5),(2.5, 0.5), (4.5, 0.5)]
# ]
# scene = sc.create_scene_from_specs(testList)
# with open('../../Week6/SmallExample', 'wb') as f:
#     pickle.dump(testList,f)
# print(scene.scene.bounds)
# print(scene.scene)
####

# # see what has just been created
# scene2.display_matplotlib()
# # if you are on ipython or jupyter you can also call 
# # a.display_ipython()
# # which generates a prettier version of this image, but on a smaller scale.

# # use get_coordinates to get the coordinates of the points in the format you expect 
# # for computing your visibility graphs and whatnot
# coords = scene2.get_polygon_coordinates()


# # We then find the optimal illumination points with Ramya's code:

# In[ ]:

from importlib import reload
import gridLP
import visibilityGraph as vs
reload(gridLP)
from gridLP import get_grid_points, get_vs_graphs, solve_lp, get_grid_points_res


import networkx as nx


# In[ ]:

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import math
from klampt.plan.cspace import CSpace,MotionPlan
from klampt.vis.glprogram import GLProgram
from klampt.math import vectorops

def interpolate(a,b,u):
    """Interpolates linearly between a and b"""
    return vectorops.madd(a,vectorops.sub(b,a),u)

class Circle:
    def __init__(self,x=0,y=0,radius=1):
        self.center = (x,y)
        self.radius = radius
        
    def contains(self,point):
        return (vectorops.distance(point,self.center) <= self.radius)

    def drawGL(self,res=0.01):
        numdivs = int(math.ceil(self.radius*math.pi*2/res))
#         print(numdivs)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(*self.center)
        for i in range(numdivs+1):
            u = float(i)/float(numdivs)*math.pi*2
            glVertex2f(self.center[0]+self.radius*math.cos(u),self.center[1]+self.radius*math.sin(u))
        glEnd()

        
class Pgon:
    def __init__(self,poly,y_bound):
        self.poly = poly
        self.y_bound = y_bound
    def contains(self,point):
        return self.poly.contains(Point(point[0],y_bound-point[1]))
    def drawGL(self):
        glBegin(GL_TRIANGLE_FAN);
        for i in list(self.poly.exterior.coords):
            glVertex2f(i[0],self.y_bound-i[1])
        glEnd()

class CircleObstacleCSpace(CSpace):
    def __init__(self,x_bound =3.0,y_bound=3.0):
        CSpace.__init__(self)
        #set bounds
        self.bound = [(0.0,x_bound),(0.0,y_bound)]
        #set collision checking resolution
        self.eps = 1e-3
        #setup a robot with radius 0.05
        self.robot = Circle(0,0,0.05)
        #set obstacles here
        self.obstacles = []

    def addObstacle(self,circle):
        self.obstacles.append(circle)
    
    def feasible(self,q):
        #bounds test
        if not CSpace.feasible(self,q): return False
        #TODO: Problem 1: implement your feasibility tests here
        #currently, only the center is checked, so the robot collides
        #with boundary and obstacles
        for o in self.obstacles:
            if o.contains(q): return False
        return True

    def drawObstaclesGL(self):
        glColor3f(0.2,0.2,0.2)
        for o in self.obstacles:
            o.drawGL()

    def drawRobotGL(self,q):
        glColor3f(0,0,1)
        newc = vectorops.add(self.robot.center,q)
        c = Circle(newc[0],newc[1],self.robot.radius)
        c.drawGL()



class CSpaceObstacleProgram(GLProgram):
    def __init__(self,space,start=(0.1,0.5),goal=(0.9,0.5),x_bound = 1.0,y_bound = 1.0,milestones = (0,0),initial_points = 1000):
        GLProgram.__init__(self)
        self.space = space
        #PRM planner
        MotionPlan.setOptions(type="prm",knn=10,connectionThreshold=1,ignoreConnectedComponents = True)
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
        #FMM* planner
        #MotionPlan.setOptions(type="fmm*")
        #self.optimizingPlanner = True
        
        #RRT planner
        #MotionPlan.setOptions(type="rrt",perturbationRadius=0.25,bidirectional=True)
        #self.optimizingPlanner = False

#         RRT* planner
#         MotionPlan.setOptions(type="rrt*")
#         self.optimizingPlanner = True
        
        #random-restart RRT planner
        #MotionPlan.setOptions(type="rrt",perturbationRadius=0.25,bidirectional=True,shortcut=True,restart=True,restartTermCond="{foundSolution:1,maxIters:1000}")
        #self.optimizingPlanner = True

        #OMPL planners:
        #Tested to work fine with OMPL's prm, lazyprm, prm*, lazyprm*, rrt, rrt*, rrtconnect, lazyrrt, lbtrrt, sbl, bitstar.
        #Note that lbtrrt doesn't seem to continue after first iteration.
        #Note that stride, pdst, and fmt do not work properly...
        #MotionPlan.setOptions(type="ompl:rrt",suboptimalityFactor=0.1,knn=10,connectionThreshold=0.1)
        #self.optimizingPlanner = True
        # we then add start, goal and milestones
        self.start=start
        self.goal=goal
#         self.planner.setEndpoints(start,goal)
        # we now add each of the chosen points as a milestone:
        self.G = self.planner.getRoadmap()
        self.start_milestones = len(self.G[0])
        print(self.start_milestones)
        for milestone in self.milestones:
            self.planner.addMilestone(milestone)
        self.components = int(self.planner.getStats()['numComponents'])
        print(self.components)
        self.G = self.planner.getRoadmap()
        self.end_milestones = len(self.G[0])
        print(self.end_milestones)
#         self.planner.addMilestone(self.start)
#         self.planner.addMilestone(self.goal)
        self.path = []
        self.G = None
        
    def keyboardfunc(self,key,x,y):
        if key==' ':
            if ((self.optimizingPlanner or not self.path) or (self.components >1)):
                print( "Planning 1...")
                self.planner.planMore(1)
                self.path = self.planner.getPath()
                self.G = self.planner.getRoadmap()
                self.components = int(self.planner.getStats()['numComponents'])
                print(self.components)

                self.refresh()
        elif key=='p':
            if ((self.optimizingPlanner or not self.path) or (self.components > 1)):
                print( "Planning 100...")
                self.planner.planMore(1000)
                self.path = self.planner.getPath()
                self.G = self.planner.getRoadmap()
                self.components = int(self.planner.getStats()['numComponents'])
                print(self.components)
                self.paths = []
#                 for i in range(self.start_milestones,self.end_milestones):
#                     for j in range(i,self.end_milestones):
#                         print('getting paths')
#                         self.paths.append( self.planner.getPath(i,j))
                self.refresh()
#         elif key=='g':
#             adjacency_matrix = np.zeros(shape = (len(self.milestones),len(self.milestones)))
#             adjacency_matrix[:,:] = np.inf
#             if(self.components == 1):
#                 for i,milestone1 in tqdm(enumerate(range(self.start_milestones+1,len(self.milestones)))):
#                     for j,milestone2 in enumerate(range(milestone1+1,len(self.milestones))):
#                         path = self.planner.getPath(milestone1,milestone2)
#                         cost = self.planner.pathCost(path)
#                         adjacency_matrix[i,j] = cost
#                         adjacency_matrix[j,i] = cost
#             print('calculated all distances')
#             return adjacency_matrix

    def display(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0,self.x_bound,self.y_bound,0,-1,1);
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glDisable(GL_LIGHTING)
        self.space.drawObstaclesGL()
        if((self.path) and (self.components == 1)):
            self.paths = []
            for i in range(self.start_milestones,self.end_milestones):
                for j in range(i+1,self.end_milestones):
                    print('getting paths')
                    self.paths.append( self.planner.getPath(i,j))
            #draw path
#             glColor3f(0,1,0)
#             glBegin(GL_LINE_STRIP)
            self.colors = []
            for i,path in enumerate(self.paths):
                if(len(self.colors) < i + 1):
                    self.colors.append([np.random.rand(),np.random.rand(),np.random.rand()])
                glColor3f(*self.colors[i])
                glBegin(GL_LINE_STRIP)
                for q in path:
                    glVertex2f(q[0],q[1])
                glEnd()
            

#             for path in self.paths:
#                 for q in path:
#                     self.space.drawRobotGL(q)
            for milestone in self.milestones:    
                self.space.drawRobotGL(milestone)

        else:
            for milestone in self.milestones:    
                self.space.drawRobotGL(milestone)
            pass

        if self.G:
            #draw graph
            V,E = self.G
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)
            glColor4f(0,0,0,0.5)
            glPointSize(3.0)
            glBegin(GL_POINTS)
            for v in V:
                glVertex2f(v[0],v[1])
            glEnd()
            glColor4f(0.5,0.5,0.5,0.5)
            glBegin(GL_LINES)
            for (i,j) in E:
                glVertex2f(V[i][0],V[i][1])
                glVertex2f(V[j][0],V[j][1])
            glEnd()
            glDisable(GL_BLEND)
    
if __name__=='__main__':
    pass
#     space = None
#     start = None
#     goal = None
#     x_bound = 1.1*scene.scene.bounds[2]
#     y_bound = 1.1*scene.scene.bounds[3]
#     space = CircleObstacleCSpace(x_bound,y_bound)

#     for poly in list(scene.scene.geoms):
#         space.addObstacle(Pgon(poly,y_bound))

#     start=(1.0,y_bound-0.5)
#     goal=(1.75,y_bound-2.25)
#     milestones = []
#     for i in chosenPoints:
#         milestones.append([i[0][0],y_bound-i[0][1]])
# #     milestone = (2.75,y_bound-0.05)
# #     start = (1.0,0.0)
# #     goal = (0,0)
#     program = CSpaceObstacleProgram(space,start,goal,x_bound = x_bound,y_bound = y_bound,
#                                     milestones = milestones, initial_points= 100)
#     program.view.w = program.view.h = 640
#     program.name = "Motion planning test"
#     adjacency_matrix = program.run()
    

# adjacency_matrix = np.zeros(shape = (len(milestones),len(milestones)))
# adjacency_matrix[:,:] = np.inf
# # if(self.components == 1):
# for i,milestone1 in tqdm(enumerate(range(program.start_milestones,program.start_milestones+1 + len(program.milestones)))):
#     for j,milestone2 in enumerate(range(milestone1+1,program.start_milestones + len(program.milestones))):
#         print(program.G[0][milestone1],program.G[0][milestone2])
#         j = j+i + 1
#         print(i,j)
#         path = program.planner.getPath(milestone1,milestone2)
#         cost = program.planner.pathCost(path)
#         adjacency_matrix[i,j] = cost
#         adjacency_matrix[j,i] = cost
        
# #         print((i,j),(j,i))
# print('calculated all distances')
# # return adjacency_matrix
# for i in range(adjacency_matrix.shape[0]):
#     adjacency_matrix[i,i] = 0



# # Motion Planner without the GUI

# In[ ]:

import math
from klampt.plan.cspace import CSpace,MotionPlan
from klampt.vis.glprogram import GLProgram
from klampt.math import vectorops


def interpolate(a,b,u):
    """Interpolates linearly between a and b"""
    return vectorops.madd(a,vectorops.sub(b,a),u)

class Circle:
    def __init__(self,x=0,y=0,radius=1):
        self.center = (x,y)
        self.radius = radius
        
    def contains(self,point):
        return (vectorops.distance(point,self.center) <= self.radius)

        
class Pgon:
    def __init__(self,poly,y_bound):
        self.poly = poly
        self.y_bound = y_bound
    def contains(self,point):
        return self.poly.contains(Point(point[0],point[1]))

class CircleObstacleCSpace(CSpace):
    def __init__(self,x_bound =3.0,y_bound=3.0):
        CSpace.__init__(self)
        #set bounds
        self.bound = [(0.0,x_bound),(0.0,y_bound)]
        #set collision checking resolution
        self.eps = 1e-3
        #setup a robot with radius 0.05
        self.robot = Circle(0,0,0.05)
        #set obstacles here
        self.obstacles = []

    def addObstacle(self,circle):
        self.obstacles.append(circle)
    
    def feasible(self,q):
        #bounds test
        if not CSpace.feasible(self,q): return False
        #TODO: Problem 1: implement your feasibility tests here
        #currently, only the center is checked, so the robot collides
        #with boundary and obstacles
        for o in self.obstacles:
            if o.contains(q): return False
        return True



class CSpaceObstacleSolver:
    def __init__(self,space,x_bound = 1.0,y_bound = 1.0,milestones = (0,0),initial_points = 4000):
        self.space = space
        #PRM planner
        MotionPlan.setOptions(type="prm",knn=10,connectionThreshold=0.05)
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
#         self.planner.addMilestone(self.start)
#         self.planner.addMilestone(self.goal)
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

        #         print((i,j),(j,i))
        print('calculated all distances')
        for i in range(self.adjacency_matrix.shape[0]):
            self.adjacency_matrix[i,i] = 0
            
        return self.adjacency_matrix





