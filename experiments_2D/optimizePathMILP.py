# from dijkstar import Graph, find_path
import sys

sys.path.append("../")


import gurobipy as gp
from gurobipy import GRB
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
import scipy.sparse as sp
from gridLP import get_grid_points_res
from experiments_2D.prm_calculator import CSpaceObstacleSolver1,CircleObstacleCSpace,MultiPgon
from shapely.geometry import Point, LineString, MultiLineString, Polygon, MultiPolygon
import time
from tqdm import tqdm
from experiments_2D.scene_creator import Scene_Creator
from planning.getToursAndPaths import getFinalPath
from experiments_2D.gridLP import compute_flux_pseudo_3d, examplePolygonList, exampleObstacleNodes, exampleEdges, get_grid_points_res, get_vs_graphs, get_irradiation_matrix, get_scene_details
import experiments_2D.plotSolutions as myplot
import experiments_2D.visibilityGraph as vs
import pdb
import pandas as pd


def getGridLines(x_res, y_res, gridPointList, obstacles):
    '''
        Returns a MultiLineString object which consists of all grid lines for a given list of grid points 
        which are viable (i.e. are entirely in free space) given a set of obstacles
    '''
    coords = list()
    for point in gridPointList:
        rightRounded = np.around(point[0] + x_res, decimals=4)
        downRounded = np.around(point[1] - y_res, decimals=4)
        rightPoint = (rightRounded, point[1])
        downPoint = (point[0], downRounded)
        rightLine = LineString([point, rightPoint])
        downLine = LineString([point, downPoint])
        rightDist = rightLine.distance(obstacles)
        downDist = downLine.distance(obstacles)
        if rightDist > 0 and rightPoint in gridPointList:
            coords.append((point, rightPoint))
        if downDist > 0 and downPoint in gridPointList:
            coords.append((point, downPoint))
    completeGrid = MultiLineString(coords)
    validGrid = completeGrid
    #validGrid = completeGrid.difference(obstacles)
    return validGrid

def getGridEdges(grid, gridPointList):
    '''
        From a given MultiString grid object, returns a list of the edges, where each edge is listed as a tuple (i,j)
        where i and j are indices referring to values in gridPointList and also returns a dictionary mapping nodes to their
        corresponding indices
    '''
    nodeToIndexDict = dict()
    for i in range(len(gridPointList)):
        nodeToIndexDict[gridPointList[i]] = i
    gridEdges = list()
    for line in list(grid.geoms):
        currCoords = list(line.coords)
        gridEdges.append((nodeToIndexDict[currCoords[0]],nodeToIndexDict[currCoords[1]]))
    return gridEdges, nodeToIndexDict

def getAdjacencyMatrixFromEdges(gridEdges, gridPointList, chosenPointIndices, allowInf=False):
    '''
        From a list of edges that connect points in the grid, we find the shortest path through the grid
        between some set of grid points in the grid (given from their indices in gridPointList)
        Arguments: gridEdges: list of feasible edges in the graph of the grid (values are tuples of the indices of points in gridPointList) 
                   gridPointList: list of nodes in the grid
    '''

    graph = nx.Graph()
    for i in range(len(gridPointList)):
        graph.add_node(i)
    for val in gridEdges:
        # Getting the end points of the edges to calculate their lengths
        i1 = val[0]
        i2 = val[1]
        p1 = Point(gridPointList[i1])
        p2 = Point(gridPointList[i2])
        dist = p1.distance(p2)
        # All the graph edges go in both directions
        graph.add_edge(i1, i2)#, weight=dist)
        graph.add_edge(i2, i1)#), weight=dist)
    # Now we find shortest paths between all pairs of points
    numPoints = len(chosenPointIndices)
    adjacencyDict = nx.all_pairs_shortest_path_length(graph)
    adjacencyMatrix = np.full((numPoints, numPoints), np.inf)
    pathDict = nx.all_pairs_shortest_path(graph)
    actualAdjacencyDict = dict()
    for val in adjacencyDict:
        actualAdjacencyDict[val[0]] = val[1]
    actualPathDict = dict()
    for val in pathDict:
        actualPathDict[val[0]] = val[1]
    for i in range(len(chosenPointIndices)):
        for j in range(i, len(chosenPointIndices)):
            gridIndexI = chosenPointIndices[i]
            gridIndexJ = chosenPointIndices[j]
            try:
                adjacencyMatrix[i,j] = adjacencyMatrix[j,i] = 0.5 * actualAdjacencyDict[gridIndexI][gridIndexJ]
            except KeyError: 
                print('something is not connected')
                if allowInf == False:
                    # If infinity values are not allowed in the adjacency matrix, then we set all 
                    # np.inf values 
                    adjacencyMatrix[i,j] = adjacencyMatrix[j,i] = 10**5   
    return adjacencyMatrix, actualPathDict

def solveNumEdgesMaximization(polygonList, chosenPoints, maxTime, minFluxReqd, pseudo_3D = False):
    '''
        Given a environment filled with surfaces to irradiate and a list of potential vantage points, 
        this function will solve a mixed integer linear program which maximizes the number of surfaces
        which are irradiated sufficiently, while ensuring that the total dwell time summed across all points
        is bounded by a given value
        Arguments: polygonList: a list which defines the obstacles in the 2D environment
                   chosenPoints: the list of potential vantage points at which the robot can stop
                   maxTime: the maximum allowed dwell-time (summed across all dwell points)
                   minFluxReqd: the minimum flux required per unit length for a surface to be irradiated sufficiently
                   pseudo_3D: Boolean value; if True, we simulate a pseudo-3D environment where each polygon has a height, but the robot only moves in a plane
    '''
    edgeList, obstacleNodeList, obstacles = get_scene_details(polygonList)
    numEdges = len(edgeList)
    numPoints = len(chosenPoints)
    
    try:
        # Create a new model
        m = gp.Model("MaximizeNumEdgesIrradiated")
        # Creating variables
            # timeVarDict variables are named from 'C0' to 'CN', where N=numPoints-1
        timeVarDict = m.addVars(numPoints, vtype=GRB.CONTINUOUS)
            # edgeVarDict variables are named from 'CN1' to CN2' where N1=numPoints and N2=numPoints+numEdges-1
        edgeVarDict = m.addVars(numEdges, vtype=GRB.BINARY)
        # Setting objective
        obj = gp.LinExpr(0.0)
        for i in range(len(edgeVarDict)):
            obj.add(edgeVarDict[i], 1.0)
        m.setObjective(obj, GRB.MAXIMIZE)
        # Setting constraints
            # Bounding total dwell-time
        timeSum = gp.LinExpr(0.0)
        for i in range(len(timeVarDict)):
            timeSum.add(timeVarDict[i], 1.0)
        m.addConstr(timeSum <= maxTime, "Total dwell-time constraint")
            # Each time variable must be non-negative
        for i in range(len(timeVarDict)):
            m.addConstr(timeVarDict[i] >= 0.0, "Sign constraint" + str(i))
            # Adding constraints for flux value for each edge
        irradiationMatrix = get_irradiation_matrix(edgeList,obstacleNodeList,chosenPoints,obstacles,height=2,power=80,pseudo_3D=pseudo_3D)
        for i in range(numEdges):
            currFluxConstr = gp.LinExpr(0.0)
            currFluxConstr.add(edgeVarDict[i], minFluxReqd)
            for j in range(numPoints):
                currFluxConstr.add(timeVarDict[j], irradiationMatrix[i][j])
            m.addConstr(currFluxConstr <= 0, "Edge flux constraint" + str(i))

        # Optimizing model
        m.optimize()

        variableValues = list()
        # Getting solution
        for v in m.getVars():
            # print('%s %g' % (v.varName, v.x))
            variableValues.append(v.x)
        timeValues = variableValues[:numPoints]
        edgeValues = variableValues[numPoints:]
        print('Obj: %g' % m.objVal)
        return timeValues, edgeValues, irradiationMatrix, m.objVal

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
    except AttributeError:
        print('Encountered an attribute error')

def solveIrradiationMaximization(polygonList, chosenPoints, maxTime, minFluxReqd, pseudo_3D = False):
    '''
        Given a environment filled with surfaces to irradiate and a list of potential vantage points, 
        this function will solve a linear program which maximizes the total flux received by all surfaces 
        (including those which are not irradiated sufficiently), while ensuring that the total dwell time 
        (summed across all dwell points) is bounded by a given value
        Arguments: polygonList: a list which defines the obstacles in the 2D environment
                   chosenPoints: the list of potential vantage points at which the robot can stop
                   maxTime: the maximum allowed dwell-time (summed across all dwell points)
                   minFluxReqd: the minimum flux required per unit length for a surface to be irradiated sufficiently
                   pseudo_3D: Boolean value; if True, we simulate a pseudo-3D environment where each polygon has a height, but the robot only moves in a plane    
    '''
    edgeList, obstacleNodeList, obstacles = get_scene_details(polygonList)
    numEdges = len(edgeList)
    numPoints = len(chosenPoints)

    try:
        # Create a new model
        m = gp.Model("MaximizeTotalFluxPercentage")
        # Creating variables
            # timeVarDict variables are named from 'C0' to 'CN', where N=numPoints-1
        timeVarDict = m.addVars(numPoints, vtype=GRB.CONTINUOUS)
            # edgeVarDict variables are named from 'CN1' to CN2' where N1=numPoints and N2=numPoints+numEdges-1
        edgeVarDict = m.addVars(numEdges, vtype=GRB.CONTINUOUS)
        # Setting objective
        obj = gp.LinExpr(0.0)
        for i in range(len(edgeVarDict)):
            obj.add(edgeVarDict[i], 1.0)
        m.setObjective(obj, GRB.MAXIMIZE)
        # Setting constraints
            # Bounding total dwell-time
        timeSum = gp.LinExpr(0.0)
        for i in range(len(timeVarDict)):
            timeSum.add(timeVarDict[i], 1.0)
        m.addConstr(timeSum <= maxTime, "Total dwell-time constraint")
            # Each time variable must be non-negative
        for i in range(len(timeVarDict)):
            m.addConstr(timeVarDict[i] >= 0.0, "Sign constraint" + str(i))
            # Percentage of flux received by each edge must be between 0 and 1 (inclusive)
        for i in range(len(edgeVarDict)):
            m.addConstr(edgeVarDict[i]>= 0, "Flux sign constraint" + str(i))
            m.addConstr(edgeVarDict[i] <= 1.0, "Upperbound flux constraint" + str(i))
        irradiationMatrix = getIrradiationMatrix(edgeList, obstacleNodeList, chosenPoints, obstacles, pseudo_3D=pseudo_3D)
            # Adding constraints for flux value for each edge
        for i in range(numEdges):
            currFluxConstr = gp.LinExpr(0.0)
            currFluxConstr.add(edgeVarDict[i], minFluxReqd)
            for j in range(numPoints):
                currFluxConstr.add(timeVarDict[j], irradiationMatrix[i][j])
            m.addConstr(currFluxConstr <= 0, "Edge flux constraint" + str(i))

        # Optimizing model
        m.optimize()

        variableValues = list()
        # Getting solution
        for v in m.getVars():
            #print('%s %g' % (v.varName, v.x))
            variableValues.append(v.x)
        timeValues = variableValues[:numPoints]
        edgeValues = variableValues[numPoints:]
        print('Obj: %g' % m.objVal)
        return timeValues, edgeValues, irradiationMatrix, m.objVal

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
    except AttributeError:
        print('Encountered an attribute error')

def solveTotalTimeMILP(polygonList, validGrid, velocity, minFluxReqd, pseudo_3D=False,scene = []):
    '''
        Given a environment filled with surfaces to irradiate and a list of potential vantage points, 
        this function will solve a linear program which minimizes the total time taken to irradiate the room
        (the total dwell-time plus the time taken to traverse the tour through all dwell-points) while 
        ensuring that each surface receives a sufficient amount of radiation to be disinfected properly 
        Arguments: polygonList: a list which defines the obstacles in the 2D environment
                   chosenPoints: the list of potential vantage points at which the robot can stop
                   velocity: speed of the robot carrying UV light
                   minFluxReqd: the minimum flux required per unit length for a surface to be irradiated sufficiently
                   pseudo_3D: Boolean value; if True, we simulate a pseudo-3D environment where each polygon has a height, but the robot only moves in a plane    
    '''
    edgeList, obstacleNodeList, obstacles = get_scene_details(polygonList)
    vReciprocal = 1/velocity

    t1 = time.time()
    #### Things I've changed
    #gridPointList = get_grid_points_res(x_res = 0.1,y_res = 0.1,xBounds = [0,bbox[2]],yBounds = [0,bbox[3]],minDist = 0.1,obstacles = scene.scene)
    print('len validgrid = {}'.format(len(validGrid)))
    numEdges = len(edgeList)
    numPoints = len(validGrid)
    grid_array = np.zeros((len(validGrid),2))
    for index,i in enumerate(validGrid):
        grid_array[index,0] = i[0]
        grid_array[index,1] = i[1]

    # scene.display_matplotlib(block = False)

    # plt.scatter(grid_array[:,0],grid_array[:,1])
    # plt.show(block = True)
    # validGrid = getGridLines(0.5,0.5,chosenPoints,obstacles)

    # graphEdges, indexDict = getGridEdges(validGrid, chosenPoints)
    print('getting adjacency matrix')
    chosenPointIndices = [i for i in range(numPoints)]
    # adjacencyMatrix, pathDict = getAdjacencyMatrixFromEdges(graphEdges,chosenPoints,chosenPointIndices)
    space = None
    start = None
    goal = None
    x_bound = 1.1*scene.scene.bounds[2]
    y_bound = 1.1*scene.scene.bounds[3]
    space = CircleObstacleCSpace(x_bound+2,y_bound+2)

    # for poly in list(scene.scene.geoms):
    #     space.addObstacle(Pgon(poly,y_bound)) 
        
    space.addObstacle(MultiPgon(scene.scene,y_bound))
    milestones = []
    for l in validGrid:
        milestones.append(l)
    program = CSpaceObstacleSolver1(space,milestones = milestones, initial_points= 1000)
    adjacencyMatrix, pathDict = program.get_adjacency_matrix_from_milestones()
    new_adjacency = np.zeros(shape = (adjacencyMatrix.shape[0]+1,adjacencyMatrix.shape[1]+1))
    new_adjacency[:-1,:-1] = adjacencyMatrix
    adjacencyMatrix = new_adjacency
    velocityAdjacencyMatrix = vReciprocal * adjacencyMatrix
    print('\n\n NAN VALUES : \n\n = ',np.isinf(adjacencyMatrix).sum())
    print('getting irradiation matrix')
    irradiationMatrix = get_irradiation_matrix(edgeList,obstacleNodeList,validGrid,obstacles,height=2,power=80,pseudo_3D=pseudo_3D)

    try:
        # Create a new model
        m = gp.Model("MinimizeTotalTime")
        m.setParam("NodefileStart", 0.5)
        m.setParam("TimeLimit", 60*60)
        m.params.MIPGap = 0.01
        m.params.Threads = 19
        m.params.Presolve = 2
        m.params.PreSparsify = 1
        m.params.Cuts = 0
        m.params.MIPFocus = 3
        m.params.ImproveStartTime = 720
        # m.params.Method = 3
        # m.params.NodeMethod = 1
        # print('no problems so far')

        # m.params.Method = 3
        print("numPoints = {} , adjacency_shape = {}".format(numPoints,adjacencyMatrix.shape))
        # Creating variables
            # timeVarDict variables are named from 'C0' to 'CN', where N=numPoints-1
        timeVarDict = m.addMVar(numPoints, vtype=GRB.CONTINUOUS, name = 'timeVars')
            # edgeVarDict variables are named from 'CN1' to CN2' where N1=numPoints and N2=numPoints+(numPoints+1)^2 - 1
        edgeVarDict = m.addMVar((numPoints+1,numPoints+1), vtype=GRB.BINARY,name = 'edgeVars')
            # networkFlowDict variables are named from 'CN3' to 'CN4', where N3 = numPoints + (numPoints+1)^2 and N4 = numPoints + 2*(numPoints+1)^2 - 1
        networkFlowDict = m.addMVar((numPoints+1,numPoints+1), vtype=GRB.CONTINUOUS, name  = 'networkVars')

        slackVars = m.addMVar(irradiationMatrix.shape[0], vtype=GRB.CONTINUOUS, name  = 'slackVars')
        # Setting objective
        numPointOnes = np.ones(numPoints)
        obj = gp.MLinExpr()
        obj = numPointOnes @ timeVarDict
        # print('gets here')
            # Assuming the last node (index - numPoints) is a "dummy" node, whose distance from all other nodes is zero
            # The path length in the objective includes only "real" edges, not involving the dummy node. Therefore, this optimization finds the shortest path, not tour
        for i in range(numPoints):
            obj += velocityAdjacencyMatrix[:,i] @ edgeVarDict[:,i]
        obj += 100*slackVars.sum()
        print('multiplies \n\n')
        m.setObjective(obj, GRB.MINIMIZE)
        # Setting constraints
            # Setting bounds for each time variable
        numPointZeros = np.zeros(numPoints)
        TIME_UPPER_LIM = 3000000
        m.addConstr(timeVarDict >= numPointZeros, "Time_sign_constraints")
        numPointM = np.full(numPoints+1, TIME_UPPER_LIM)
        # print('gets here2 ')
        for i in range(numPoints):
            m.addConstr(timeVarDict[i] <= edgeVarDict[i,:] @ numPointM, "Time_upper-bound_constraint"+str(i))
            # Setting bounds for each network flow variable
        numPointPlusZeros = np.zeros(numPoints+1)
        for i in range(numPoints+1):
            m.addConstr(networkFlowDict[i,:] >= numPointPlusZeros, "Flow_sign_constraint" + str(i))
            ubVector = np.full(1,TIME_UPPER_LIM)
            m.addConstr(networkFlowDict[i,:] <= edgeVarDict[i,:] * TIME_UPPER_LIM, "Flow_upper-bound_constraint" + str(i))
            # No edges allowed from a vertex to itself
        for i in range(numPoints+1):
            m.addConstr(edgeVarDict[i,i] == 0, "No_loop_edges_constraint" + str(i))
            # No flow allowed from a vertex to itself
        for i in range(numPoints+1):
            m.addConstr(networkFlowDict[i,i] == 0, "No_loop_flows_constraint" + str(i))
            # Making sure tour starts at the dummy node (which has index numPoints)
        numPointPlusOnes = np.ones(numPoints+1)
        m.addConstr(numPointPlusOnes @ edgeVarDict[numPoints,:] == 1, "Start_at_dummy_node")
            # Adding constraints for required flux value for each edge
        minFluxVector = np.full(numEdges,minFluxReqd)
                # Multiply by -1 since the original irradiation matrix stores all negative values
        sparseIrradiationMatrix = sp.csr_matrix(-1 * irradiationMatrix)
        # print('gets here 3')
        m.addConstr(sparseIrradiationMatrix @ timeVarDict + slackVars>= minFluxVector, "Edge_flux_constraints")
        # ensure all slacks are greater of equal to zero
        m.addConstr(slackVars >= 0, 'slack_constraints')
            # Ensuring each selected vertex has degree 2
        # print('gets here 4')
        for i in range(numPoints+1):
            m.addConstr(numPointPlusOnes @ edgeVarDict[i,:] == numPointPlusOnes @ edgeVarDict[:,i], "Ensure_degree_2_constraint" + str(i))
            m.addConstr(numPointPlusOnes @ edgeVarDict[i,:] <= 1, "In-flux_limit_constraint" + str(i))
        # print('almost there')
            # Bounding flow values
        flowBound = m.addVar(vtype = GRB.CONTINUOUS)
        m.addConstr(flowBound == edgeVarDict.vararr.sum() - 1)
        maximum = m.addVar(vtype=GRB.CONTINUOUS)
        m.addConstr(maximum == gp.max_(networkFlowDict.vararr.flatten().tolist()))
        m.addConstr(maximum -flowBound <= 0)
            # Ensuring equal-density flow across the tour
        for i in range(numPoints):
            m.addConstr(numPointPlusOnes @ networkFlowDict[:,i] - numPointPlusOnes @ networkFlowDict[i,:] ==
                     numPointPlusOnes @ edgeVarDict[i,:], "Equal_density-flow_constraint" + str(i))
        # Optimizing model
        # m.write('..\Week9\model_matrix.lp')
        # print('Going to optimize')
        m.optimize()

        variableValues = list()
        # Getting solution
        # for v in m.getVars():
        #     print('%s %g' % (v.varName, v.x))
        #     variableValues.append(v.x)
        timeValues = timeVarDict.x.flatten()
        edgeValues = edgeVarDict.x.flatten()
        print('\n\n\n slack values = {} \n\n\n'.format(slackVars.x.sum()))
        # pdb.set_trace()
        # timeValues = m.getVarByName('timeVars').x
        # edgeValues = m.getVarByName('edgeVars').x
        print('Obj: %g' % m.objVal)
        return timeValues, edgeValues, irradiationMatrix,pathDict, m.objVal,adjacencyMatrix
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
    except AttributeError:
        print('Encountered an attribute error')

def extractTour(edgeValues, chosenPoints):
    '''
        Given a solution from the MILP, this function will return the path described by the binary edge values
        The path is actually extracted from a tour that starts at a dummy node
        Arguments: edgeValues: a list of length (n+1)^2 (where len(chosenPoints) = n), giving binary values. A value of 1 
                               means the tour goes from node i to node j (nodes defined by chosenPoints)
                   chosenPoints: the list of potential vantage points used in the MILP
    '''
    numPoints = len(chosenPoints)
    # This dictionary will collect dwell-nodes as keys and their values will be the nodes they go to next
    visitedNodesDict = dict()
    for i in range(numPoints+1):
        for j in range(numPoints+1):
            currEdge = edgeValues[(numPoints+1)*i + j]
            if currEdge == 1:
                if i == numPoints:
                    visitedNodesDict['start'] = j
                elif j == numPoints:
                    visitedNodesDict[i] = 'end'
                else:
                    visitedNodesDict[i] = j
    dictKeys = list(visitedNodesDict.keys())
    tourIncomplete = True
    tour = list()
    # Starting at the dummy node
    currNode = 'start'
    tour.append(currNode)
    while tourIncomplete:
        currNode = visitedNodesDict[currNode]
        tour.append(currNode)
        if currNode == 'end':
            tourIncomplete = False
    return tour

def getActualEdges(gridEdges,gridPointList):
    '''
        With a list of grid edges, where each tuple is a pair of indices corresponding to points in gridPointList,
        this function converts the indices to the actual nodes they represent
    '''
    actualEdgeList = list()
    for val in gridEdges:
        actualEdgeList.append((gridPointList[val[0]], gridPointList[val[1]]))
    return actualEdgeList

def main():
    room = []
    dwell_times = []
    lengths = []
    runtimes = []
    resolutions = []
    vantage = []
    for roomnum in range(0,25):
        fileName = '../data/2D_data/rooms/room_'+str(roomnum)+'/scene_'+str(roomnum)+'.p'
        with open(fileName, 'rb') as f:
            scene = pickle.load(f)
            #SceneCreator = Scene_Creator()
            #scene = SceneCreator.remake_scene_to_resolution(scene, 0.05)
            polygonList = scene.get_polygon_coordinates()
        #polygonList = examplePolygonList
        usingEdgeList, obstacleNodeList, obstacles = get_scene_details(polygonList)
        #usingEdgeList = exampleEdges
        bbox = scene.get_bbox()
        res = 1
        gridPointList = get_grid_points_res(x_res = res,y_res = res,xBounds = [bbox[0]-0.5,bbox[2]+0.5],yBounds = [bbox[1]-0.5,bbox[3]+0.5],minDist = 0.1,obstacles = scene.scene)
        startT = time.time()
        #timeValues, edgeValues, irradiationMatrix, objValue = solveNumEdgesMaximization(polygonList, gridPointList, 100, minFluxReqd=280, pseudo_3D=True)
        #endT = time.time()
        # timeValues, edgeValues, irradiationMatrix, objValue = solveIrradiationMaximization(polygonList, gridPointList, 84.4, 1.1, pseudo_3D=True)
        # testMILP()
        timeValues, edgeValues, irradiationMatrix, pathDict, objValue,adjacencyMatrix = solveTotalTimeMILP(polygonList, gridPointList, velocity=0.5, minFluxReqd=280, pseudo_3D=True,scene = scene)
        endT = time.time()
        # validGrid = getGridLines(0.5,0.5,gridPointList,obstacles)
        # graphEdges, indexDict = getGridEdges(validGrid, gridPointList)
        # gridEdges = getActualEdges(graphEdges, gridPointList)
        print('edgeValues = {} , gridPointList = {}'.format(len(edgeValues),len(gridPointList)))
        # tour = extractTour(edgeValues, gridPointList)
        # print(tour)
        # print([gridPointList[i] for i in tour[1:-1]])
        gridPointListIndices = [i for i in range(len(gridPointList))]
        # pathMilestones, pathEdges, pathLength = getFinalPath(tour[1:-1], pathDict, gridPointList, gridPointListIndices)
        totalTime = 0
        vantagePoints = list()
        vantagePointTimes = list()
        euclideanTour = list()
        pathLength = np.dot(edgeValues.flatten(),adjacencyMatrix.flatten())
        # for i in range(1,len(tour)-1):
        #     currVal = timeValues[tour[i]]
        #     if currVal > 0:
        #         euclideanTour.append(gridPointList[tour[i]])
        for i in range(len(gridPointList)):
            currVal = timeValues[i]
            if currVal > 0:
                print('Grid point ', gridPointList[i], ' has time ', currVal)
                vantagePoints.append(gridPointList[i])
                vantagePointTimes.append((gridPointList[i], currVal))
                totalTime += currVal
        # euclideanTour = [gridPointList[tour[i]] for i in range(1, len(tour) - 1)]
        print('No. of edges irradiated is: ', objValue)
        print('Dwell-time of solution is: ', totalTime)
        # print('Disinfection time of solution is: ', objValue)
        print('Path-length of solution is: ', pathLength)
        print('No. of dwell-points is: ', len(vantagePoints))
        print('Runtime of solution is: ', endT - startT)
        # myplot.plotGrid(gridPointList)
        # myplot.plotGridLines(gridEdges,"green")
        # myplot.plotChosenPoints(euclideanTour, label=True)
        # myplot.plotPointTimes(vantagePointTimes)
        # myplot.plotObstacles(pathEdges, "red")
        # myplot.plotEdgeFlux(usingEdgeList, irradiationMatrix, timeValues)
        # myplot.show()
        dwell_times.append(totalTime)
        lengths.append(pathLength)
        resolutions.append(res)
        runtimes.append(endT - startT)
        vantage.append(len(vantagePoints))
    df = pd.DataFrame({"dwell_times":dwell_times,"path_lengths":lengths,"resolution":resolutions,"runtimes":runtimes,"vantage_points":vantage})
    df.to_csv('MILP_experiments_25cm.csv', sep = '|', index = False)

if __name__ == "__main__":
    main()

