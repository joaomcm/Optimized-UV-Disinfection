from scene_creator import Scene_Creator
import numpy as np
import pickle
import time
import pandas as pd
from gridLP import get_grid_points, get_vs_graphs, solve_lp, get_grid_points_res,get_internal_grid_points_res
from shapely.geometry import Polygon, LineString, LinearRing, MultiPolygon,Point,MultiPoint 
from importlib import reload
from shapely.geometry.collection import GeometryCollection
import plotSolutions
reload(plotSolutions)
import plotSolutions as myplot
import gridLP
from gridLP import solve_lp_gurobi, get_irradiation_matrix
import visibilityGraph as vs
reload(gridLP)
from gridLP import get_grid_points, get_vs_graphs, solve_lp, get_grid_points_res, get_scene_details, get_midpoint_grid_points
from matplotlib import pyplot as plt
from prm_calculator import CSpaceObstacleSolver1,CircleObstacleCSpace,MultiPgon
from optimizePathMILP import getGridLines, getGridEdges, getActualEdges, getAdjacencyMatrixFromEdges
from getToursAndPaths import getTour, readTourFile, getPathFromPrm, getFinalPath
import os
import pdb
from tqdm import tqdm 

time_matrix = np.zeros((26,5),dtype='float64')
time_matrix[0][0] = 0 # 'Room Number'
time_matrix[0][1] = 0 # 'Time taken to solve the LP (including getting grid points, finding visibility graphs, and solving the LP)'
time_matrix[0][2] = 0 # 'Dwell time of solution'
time_matrix[0][3] = 0 # 'Number of dwell points'
time_matrix[0][4] = 0 

def solveScene(fileName, results_dir = '../Media/Complete_Pipeline_2D',res = 0.5,env_resolution = 1):
# ## Start the pipeline as usual - by creating or loading a scene and optimizing its viewpoints

    sc = Scene_Creator()
    # scene = sc.create_scene(total_polygons = 10,max_x =4.0,max_y = 4.0,max_scale = 1.5,max_tries = 100,max_skew = 10,max_rot = 60, min_clearance = 0.3,resolution = 1000 )
    # poly = scene.scene.geoms[0]

    scene = sc.load_scene(fileName)

    # scene.display_matplotlib()

    scene2 = sc.remake_scene_to_resolution(scene,env_resolution)
    # scene2.display_matplotlib()

    #### Things I've changed
    #gridPointList = get_grid_points_res(x_res = 0.1,y_res = 0.1,xBounds = [0,bbox[2]],yBounds = [0,bbox[3]],minDist = 0.1,obstacles = scene.scene)
    bbox = scene2.get_bbox()
    t1 = time.time()
    #### Things I've changed
    #gridPointList = get_grid_points_res(x_res = 0.1,y_res = 0.1,xBounds = [0,bbox[2]],yBounds = [0,bbox[3]],minDist = 0.1,obstacles = scene.scene)
    gridPointList = get_grid_points_res(x_res = res,y_res = res,xBounds = [bbox[0]-0.5,bbox[2]+0.5],yBounds = [bbox[1]-0.5,bbox[3]+0.5],minDist = 0.1,obstacles = scene2.scene)

    # print('\n\n\n\n RESOLUTION WITHIN THE PROGRAM {} - Len of gridPointList {} \n\n\n'.format(res,len(gridPointList)))

    # polygonList = scene.get_polygon_coordinates()
    # gridPointList,_ = get_midpoint_grid_points(polygonList, minDist = 0.1)
    # print(gridPointList)
    # ## We then find the optimal illumination points with Ramya's code:
    bbox = scene.get_bbox()


    # ## checking if my gridding function makes sense
    k = []
    for i in gridPointList:
        k.append(Point(i[0],i[1]))
    c = MultiPoint(k)
    # c
    t2 = time.time()
    # looks reasonable. Let's now find the visibility graphs:


    # ## getting visibility graphs:

    vsGraphList = get_vs_graphs(gridPointList,scene2.get_vertex_coordinates(),scene2.get_edges(),use_shapely = True,obstacles = scene2.scene)
    # ## we then solve the LP


    solution, A_ub,_ = solve_lp_gurobi(scene2.get_polygon_coordinates(),gridPointList,280,
                            height = 2, power = 80 ,pseudo_3D = True, interior = False,vsGraphList = vsGraphList)

    timeSum = 0
    chosenPoints = []
    for i in range(len(gridPointList)):
            currVal = solution[i]
            timeSum += currVal
            if currVal > 0:  # Can make the if statement 'if currVal > 0' when using 'simplex' or 'revised-simplex'
                # print('Grid Point ', gridPointList[i], ' has time ', currVal)
                chosenPoints.append((gridPointList[i], currVal))
    t3 = time.time()
    print('\n\n\n\n {} \n\n\n '.format(timeSum))

    ### UNCOMMENT EVERYTHING AFTER THIS ONCE FINISHED
    # # print('No. of grid points: ', len(gridPointList))
    # # print('Solution time: ',timeSum)
    # # print('The number of chosen points is: ', len(chosenPoints))
    # # print('Time taken to attain solution is: ', endT - startT)
    # # Plotting 
    # edgeList, obstacleNodeList, obstacles = get_scene_details(polygonList)
    # myplot.plotGrid(gridPointList)
    # myplot.plotObstacles(edgeList, 'blue')
    # myplot.plotPointTimes(chosenPoints)
    # myplot.plotEdgeFlux(scene.get_edges(), A_ub, solution)
    # myplot.show()
    # scene2.display_matplotlib()

    # # ## we then instantiate the problem with the PRM planner:
    space = None
    start = None
    goal = None
    x_bound = 1.1*scene.scene.bounds[2]
    y_bound = 1.1*scene.scene.bounds[3]
    space = CircleObstacleCSpace(x_bound,y_bound)

    # for poly in list(scene.scene.geoms):
    #     space.addObstacle(Pgon(poly,y_bound)) 
        
    space.addObstacle(MultiPgon(scene.scene,y_bound))

    milestones = []
    for i in chosenPoints:
        milestones.append([i[0][0],i[0][1]])
    # program = CSpaceObstacleSolver(space,x_bound = x_bound,y_bound = y_bound,
    #                                 milestones = milestones, initial_points= 2000)
    # adjacency_matrix, pathDict = program.get_adjacency_matrix_from_milestones()
    program = CSpaceObstacleSolver1(space,milestones = milestones, initial_points= 200)
    adjacency_matrix, pathDict = program.get_adjacency_matrix_from_milestones()

    adjacency_matrix *= 10000 # To get it to work for TSP
    # # you can also obtain the graph to calculate the paths once you obtain the TSP with:
    t4 = time.time()

    PRM_graph = program.G

    # and thus we conclude our business! 

    print(program.G)
    # # With the distance matrix, we run a TSP solver to find a good tour through the dwell points
    getTour(adjacency_matrix, 'currTSP') # We just have an arbitrary name since it doesn't matter - can change this so that user can input filename if desired
    tour = readTourFile('currTSP.txt', milestones)
    # Now, we can convert this tour (described by the indices of the milestones) into a tour 
    # described by Euclidean co-ordinates (the original co-ordinates, from chosenPoints)
    euclideanTour = [milestones[tour[i]] for i in range(len(tour))]
    path_length = 0
    # pdb.set_trace()
    print(tour,adjacency_matrix.shape)
    for i in range(len(tour[:-1])):
        # print(adjacency_matrix[tour[i],tour[i+1]])
        path_length += adjacency_matrix[tour[i],tour[i+1]]/10000
    # # Plotting the TSP tour - the order in which to traverse the milestones - this doesn't define the ACTUAL paths to take
    polygonList = scene.get_polygon_coordinates()
    edgeList, obstacleNodeList, obstacles = get_scene_details(polygonList)
    myplot.plotChosenPoints(euclideanTour)
    myplot.plotObstacles(edgeList, "blue")
    actualPath = list()
    for i in range(len(euclideanTour)-1):
        actualPath.append((euclideanTour[i], euclideanTour[i+1]))
    myplot.plotObstacles(actualPath, "red")
    # myplot.show()
    # print(tour)
    # print(milestones)
    # print(euclideanTour)

    # # Finding optimal paths for each milestone-to-milestone pair in the tour
    # finalPath = getPathFromPrm(tour,pathDict,milestones, polygonList, minDist=0.1)
    # t5 = time.time()
    # # Plotting again - the actual final path
    # myplot.clear()
    # myplot.plotChosenPoints(euclideanTour, label=False)
    # myplot.plotObstacles(edgeList, "blue")
    # actualPath = list()
    # for i in range(len(finalPath)-1):
    #     actualPath.append((finalPath[i], finalPath[i+1]))
    # myplot.plotEdgeFlux(edgeList, A_ub, solution)
    # myplot.plotObstacles(actualPath, "red")
    # # myplot.show()
    # if(not os.path.exists(results_dir)):
    #     os.mkdir(results_dir)
    # myplot.savefig(results_dir+'/'+os.path.split(fileName)[-1].split('.')[0]+'.pdf')
    # vantagePointSelection = t2 - t1
    # lpSolveTime = t3 - t2
    # prmSolverTime = t4 - t3
    # TSPSolveTime = t5 - t4
    #solveTime = endT - startT
    return np.sum(solution),path_length,len(tour)


# Chose pickled scene to work with
# fileName = 'perfect_scene_low_res.p'
# solveScene(fileName)



room = []
dwell_times = []
lengths = []
runtimes = []
resolutions = []
vantage = []
env_resolutions = []

for res in tqdm([1/(2**j) for j in range(5,8)]):
    for i in range(0,25):
        env_res = 1.0/8
        fileName = './adjacency_matrices/iteration_'+str(i)+'/scene_'+str(i)+'.p'
    #   if(i!= 14):
        print('\n\n\n Doing Room {} with Resolution {}\n\n\n'.format(i,res))
        start = time.time()
        dwell, path_length,points = solveScene(fileName,res = res,env_resolution = env_res)
        runtime = -start+time.time()
        dwell_times.append(dwell)
        lengths.append(path_length)
        env_resolutions.append(env_res)
        resolutions.append(res)
        vantage.append(points)
        runtimes.append(runtime)
        room.append(i)
    df = pd.DataFrame({"dwell_times":dwell_times,"path_lengths":lengths,"env_resolution":env_resolutions,"runtimes":runtimes,"vantage_points":vantage,'room':room,'resolutions':resolutions})
    df.to_csv('LP_grid_res_experiments.csv', sep = '|', index = False,mode = 'a',header = False)


    # time_matrix[i+1][0] = i
    # time_matrix[i+1][1] = vantagePointSelection
    # time_matrix[i+1][2] = lpSolveTime
    # time_matrix[i+1][3] = prmSolverTime
    # time_matrix[i+1][4] = TSPSolveTime

# timeDF = pd.DataFrame(time_matrix)
# timeDF.to_csv('../fullPipeline/results025.csv')
# solveScene('./adjacency_matrices/iteration_4/scene_4.p')