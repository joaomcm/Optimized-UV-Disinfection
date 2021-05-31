max_skew = 30
max_x = 50
max_y = 50
max_scale = 5
max_rot = 120
# !pip3 install shapely
# !pip3 install descartes
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
from experiments_2D.figures import BLUE, SIZE, set_limits, plot_coords, color_isvalid
from shapely.geometry import MultiPoint,Point
from shapely.ops import split
from tqdm import tqdm

# from importlib import reload
# import experiments_2D.gridLP
import experiments_2D.visibilityGraph as vs
# reload(gridLP)
from experiments_2D.gridLP import get_grid_points, get_vs_graphs, solve_lp, get_grid_points_res

import experiments_2D.plotSolutions as plotSolutions
# reload(plotSolutions)
import experiments_2D.plotSolutions as myplot

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
    def display_matplotlib(self, block = False):
        fig = plt.figure(1,figsize= SIZE,dpi = 90)
        ax = fig.add_subplot(111)
        for polygon in self.scene:
            plot_coords(ax,polygon.exterior)
            patch = patches.Polygon(polygon.exterior.xy, facecolor=color_isvalid(self.scene), edgecolor=color_isvalid(self.scene, valid=BLUE), alpha=0.5, zorder=2)
            ax.add_patch(patch)
        plt.show(block = block)
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
        