import gurobipy as gp
from gurobipy import GRB
import math
import numpy as np
import pickle
import time
from scipy import optimize
import scipy.sparse as sp
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
import experiments2d.visibilityGraph as vs
import experiments2d.plotSolutions as myplot
from joblib import Parallel, delayed
from tqdm import tqdm

examplePolygonList = [
[(4.5, 0.5),(4.5, 4.5),(2.5, 4.5),(2.5, 0.5), (4.5, 0.5)]
]
exampleObstacleNodes = [(4.5, 0.5),(4.5, 4.5),(2.5, 4.5),(2.5, 0.5)]
exampleEdges = [((4.5,0.5),(4.5,4.5)), ((4.5,4.5), (2.5, 4.5)), ((2.5, 4.5), (2.5, 0.5)), ((2.5,0.5), (4.5,0.5))]

examplePolygonList2 = [[(-13,2),(-13,1),(-12,1),(-12,2),(-13,2)],
         [(-11,4),(-11,-2),(-9,-2),(-9,4),(-11,4)],
         [(-8,7),(-7,6),(-6,6),(-5,7),(-6,8),(-7,8),(-8,7)],
         [(-2,8),(-2,0),(0,0),(0,8),(-2,8)],
         [(3,6),(4,6),(5,7),(4,8),(3,8),(2,7),(3,6)],
         [(11,12),(11,7),(13,7),(13,12),(11,12)],
         [(13,5),(13,2),(12,2),(12,5),(13,5)],
         [(13,-3),(11,-5),(14,-5),(13,-3)]
]
exampleObstacleNodes2 = [(-13,2),(-13,1),(-12,1),(-12,2),
                        (-11,4),(-11,-2),(-9,-2),(-9,4),
                        (-8,7),(-7,6),(-6,6),(-5,7),(-6,8),(-7,8),
                        (-2,8),(-2,0),(0,0),(0,8),
                        (3,6),(4,6),(5,7),(4,8),(3,8),(2,7),
                        (11,12),(11,7),(13,7),(13,12),
                        (13,5),(13,2),(12,2),(12,5),
                        (13,-3),(11,-5),(14,-5)]
exampleEdges2 = [((-13,2),(-13,1)), ((-13,1),(-12,1)),((-12,1),(-12,2)),((-12,2),(-13,2)),
                ((-11,4),(-11,-2)),((-11,-2),(-9,-2)),((-9,-2),(-9,4)),((-9,4),(-11,4)),
                ((-8,7),(-7,6)),((-7,6),(-6,6)),((-6,6),(-5,7)),((-5,7),(-6,8)),((-6,8),(-7,8)),((-7,8),(-8,7)),
                ((-2,8),(-2,0)),((-2,0),(0,0)),((0,0),(0,8)),((0,8),(-2,8)),
                ((3,6),(4,6)),((4,6),(5,7)),((5,7),(4,8)),((4,8),(3,8)),((3,8),(2,7)),((2,7),(3,6)),
                ((11,12),(11,7)),((11,7),(13,7)),((13,7),(13,12)),((13,12),(11,12)),
                ((13,5),(13,2)),((13,2),(12,2)),((12,2),(12,5)),((12,5),(13,5)),
                ((13,-3),(11,-5)),((11,-5),(14,-5)),((14,-5),(13,-3))]

examplePolygonList3 = [[(1.5,8.5),(1.5,9.5),(9.5,9.5),(9.5,8.5),(1.5,8.5)],
                       [(4.5, 0.5),(4.5, 4.5),(2.5, 4.5),(2.5, 0.5),(4.5,0.5)],
                       [(7.5,4),(5.5,2),(9.5,2),(7.5,4)] 
                    ]
exampleObstacleNodes3 = [(1.5,8.5),(1.5,9.5),(9.5,9.5),(9.5,8.5),
                         (4.5, 0.5),(4.5, 4.5),(2.5, 4.5),(2.5, 0.5),
                         (7.5,4),(5.5,2),(9.5,2)]

exampleEdges3 = [((1.5,8.5),(1.5,9.5)),((1.5,9.5),(9.5,9.5)),((9.5,9.5),(9.5,8.5)),((9.5,8.5),(1.5,8.5)),
                 ((4.5,0.5),(4.5,4.5)), ((4.5,4.5), (2.5, 4.5)), ((2.5, 4.5), (2.5, 0.5)), ((2.5,0.5), (4.5,0.5)),
                 ((7.5,4),(5.5,2)),((5.5,2),(9.5,2)),((9.5,2),(7.5,4))]

def get_grid_points(numRow, numCol, xBounds, yBounds, polygonList, minDist, maxDist):
    '''
        Input: numRows - the number of grid points we want across a row of the grid
               numCols - the number of grid points we want across a column of the grid
               xBounds - a tuple (leftX, rightX) which gives us the x-coordinate bounds of our space
               yBounds - a tuple (bottomY, topY) which gives us the y-coordinate bounds of our space
               polygonList - a list of lists of edges which define the obstacles in our grid space
               minDist - if x is the distance of the robot from the closest obstacle, then we want x > minDist
                         This value must be non-negative
               maxDist - if x_C is the distance of the robot from obstacle C, then to include a grid point, 
                         we must have x_C <= maxDist for at least one obstacle in polygonList
        Output: a list of grid points across our environment - this is assuming it is a rectangular space
                there should be numRows*numCols points in the space. It should only include the grid points
                that don't occur inside any of the obstacles 

        Note: Here, resolution is defined as ((xBounds[1] - xBounds[0])*(yBounds[1] - yBounds[0]))/(numRow*numCol)
              To get a square grid, we will want (xBounds[1] - xBounds[0])/numRow = (yBounds[1] - yBounds[0])/numCol
    '''

    rowGap = (xBounds[1] - xBounds[0])/(numRow - 1)
    colGap = (yBounds[1] - yBounds[0])/(numCol - 1)
    gridPointList = list()

    for i in range(numRow):
        for j in range(numCol):
            currPoint = Point((xBounds[0] + rowGap*i, yBounds[0] + colGap*j))
            tooClose = False
            # tooFar is true until we find a polygon whose distance from currPoint is less than maxDist
            tooFar = True
            for polygon in polygonList:
                currPolygon = Polygon(polygon)
                closestDistance = currPoint.distance(currPolygon)
                if closestDistance <= minDist:
                    tooClose = True
                    break
                if closestDistance <= maxDist:
                    tooFar = False
            if tooClose == False and tooFar == False:
                gridPointList.append((xBounds[0] + rowGap*i, yBounds[0] + colGap*j))
    return gridPointList

def get_grid_points_res(x_res,y_res,xBounds,yBounds,obstacles,minDist):
    x_values = np.arange(xBounds[0],xBounds[1]+x_res,x_res)
    y_values = np.arange(yBounds[0],yBounds[1]+y_res,y_res)
    x_grid, y_grid = np.meshgrid(x_values,y_values)
    print(x_grid.shape)
    x_grid = np.around(x_grid.flatten(),decimals = 4)
    y_grid = np.around(y_grid.flatten(),decimals = 4)
    gridPointList = []
    points = []
    for i in range(len(x_grid)):
        currx = x_grid[i]
        curry = y_grid[i]
        currPoint = Point(currx,curry)
        tooClose = False
        closestDistance = currPoint.distance(obstacles)
        if(closestDistance <= minDist):
            pass
        else:
            gridPointList.append((currx,curry))
    return gridPointList

def get_vs_graphs(startNodeList, edgeNodeList, edgeList,use_shapely = False,obstacles = []):
    '''
        Returns a list of the visibility graphs for each of the nodes in startNodeList, given obstacles
        defined by the edges in edgeList
    '''
    vsGraphList = list()
    # serial version
    # for node in startNodeList:
    #     currGraph = vs.generate_visibility_graph_no_rotation(node, edgeNodeList, edgeList)
    #     vsGraphList.append(currGraph)

    # parallel_version
    vsGraphList = Parallel(n_jobs = -2,verbose = 2, prefer = 'threads')(delayed(vs.generate_visibility_graph_no_rotation)(node,edgeNodeList,edgeList,use_shapely = use_shapely,obstacles = obstacles) for node in startNodeList)
    return vsGraphList

def get_irradiation_matrix(edgeList, nodeList, chosenPoints, obstacles, height=2, power=1, pseudo_3D = False,vsGraphList = []):
    '''
        Function which takes a description of a scene and a list of potential vantage points,
        and returns an irradation matrix A, where A[i,j] gives the amount of radiation per unit length
        received by edge i, when the UV light is placed at grid point j for a single unit of time.
        Arguments: edgeList: list of the edges which must be irradiated
                   nodeList: list of the nodes of the edges
                   chosenPoints: list of potential vantage points
                   obstacles: a MultiPolygon object describing all the obstacles in the scene
    '''
    if(pseudo_3D):
        print('\n\ncalculating 2.5 D fluencies for the room\n\n')
    numEdges = len(edgeList)
    numPoints = len(chosenPoints)
    if(len(vsGraphList) == 0):
        vsGraphList = get_vs_graphs(chosenPoints,nodeList,edgeList,use_shapely=True,obstacles=obstacles)
    irradiationMatrix = np.zeros((numEdges, numPoints),dtype=float) 
    for j in tqdm(range(numPoints)):
        # Finding all the obstacle vertices visible from the current grid point, and sorting them by 
        # angle subtended 
        x = chosenPoints[j]
        obstacleVertexList = [edge[1] for edge in vsGraphList[j]]
        sortedVertexList = vs.sort_nodes(x, obstacleVertexList)
        # Adding the first vertex to the end of the list to complete the cycle
        sortedVertexList.append(sortedVertexList[0])
        for k in range(len(sortedVertexList)-1):
            potentialEdge1 = (sortedVertexList[k], sortedVertexList[k+1])
            potentialEdge2 = (sortedVertexList[k+1], sortedVertexList[k])
            # Checking if either of these potential edges are actually edges
            # Note that either exactly one of them, or neither of them will be edges (i.e. the same edge
            # will not be in the list twice in different orderings)
            try: 
                edgeIndex = edgeList.index(potentialEdge1)
            except ValueError:
                try:
                    edgeIndex = edgeList.index(potentialEdge2)
                except ValueError:
                    edgeIndex = None
            if edgeIndex != None:
                # In this case, the two vertices considered form an obstacle edge AND this entire edge must be
                # visible from the current grid point
                if not pseudo_3D:
                    vantagePoint = Point(x)
                    edgePoint1 = Point(sortedVertexList[k])
                    edgePoint2 = Point(sortedVertexList[k+1])
                    oppEdgeLen = edgePoint2.distance(edgePoint1)
                    adjEdge1Len = vantagePoint.distance(edgePoint1)
                    adjEdge2Len = vantagePoint.distance(edgePoint2)
                    # Finding angle subtended by obstacle edge from vantage point, using Law of Cosine
                    acosArg = (adjEdge1Len**2 + adjEdge2Len**2 - oppEdgeLen**2)/(2*adjEdge1Len*adjEdge2Len)
                    # Guaranteeing argument is between -1 and 1 (to avoid numerical overflow errors)
                    if acosArg < -1 or acosArg > 1:
                        print('Outside bounds: ', acosArg)
                    acosArg = min(acosArg, 1.0)
                    acosArg = max(acosArg, -1.0)
                    angleSubtended = math.acos(acosArg)
                    # The ratio of this angle to 2*pi gives the section of area of the circle which the edge receives radiation from
                    irradiationMatrix[edgeIndex][j] = -angleSubtended/(2*math.pi*oppEdgeLen)
                else:
                    this_edge = [sortedVertexList[k],sortedVertexList[k+1]]
                    irradiationMatrix[edgeIndex][j] = compute_flux_pseudo_3d(this_edge,x,height = height, power = power)
    return irradiationMatrix

def get_scene_details(polygonList):
    '''
        Given a list of polygons which describe a scene, this function will return the list of
        edges, the list of edge nodes, and a MultiPolygon object describing all the polygons in the scene
        Arguments: polygonList: a list of polygons which define the obstacles in the 2D environment
    '''
    edgeList = list()
    nodeList = list()
    multiPolygonList = list()
    for polygon in polygonList:
        # Here, assuming the polygon is defined by putting the same vertex at the beginning and end of the list
        currPolygon = Polygon(polygon)
        multiPolygonList.append(currPolygon)
        for i in range(len(polygon)-1):
            edgeList.append((polygon[i], polygon[i+1]))
            nodeList.append(polygon[i])
    obstacles = MultiPolygon(multiPolygonList)

    return edgeList, nodeList, obstacles

def solve_lp(gridPointList, nodeList, edgeList,obstacles,height = 2,power = 1,pseudo_3D = False):
    '''
        This function takes a list of grid points, a list of visibility graphs for each of these grid points
        corresponding to the obstacles defined by the edges in edgeList, and solves a linear program that tries to
        minimize the total amount of time we stop at each of the points, while constraining by the amount of
        radiation each edge needs to receive
    '''
    # The constraints are defined by making sure each edge receives the desired amount of radiation
    A_ub = get_irradiation_matrix(edgeList,nodeList,gridPointList,obstacles,height=height,power=power,pseudo_3D=pseudo_3D)
    print(np.sum(A_ub < 0))
    b_ub = np.full((numConstraints,1), -1.0)
    b_ub -= 0.1 # Adding a buffer to make sure flux actually exceeds 1 for each edge
    c = np.ones((numVariables, 1))
    
    # Having defined the LP, we can solve it
    sol = optimize.linprog(c, A_ub, b_ub, method="revised simplex") # By default, the variables must take non-negative real values, which is what we want
    return A_ub, sol

def solve_lp_gurobi(polygonList, chosenPoints, minFluxReqd, height = 2,power = 1,pseudo_3D = False,interior = False,vsGraphList = []):
    '''
        Given a environment filled with surfaces to irradiate and a list of potential vantage points, 
        this function will solve a mixed integer linear program which maximizes the number of surfaces
        which are irradiated sufficiently, while ensuring that the total dwell time summed across all points
        is bounded by a given value
        Arguments: polygonList: a list which defines the obstacles in the 2D environment
                   chosenPoints: the list of potential vantage points at which the robot can stop
                   minFluxReqd: the minimum flux required per unit length for a surface to be irradiated sufficiently
    '''
    edgeList, obstacleNodeList, obstacles = get_scene_details(polygonList)
    numEdges = len(edgeList)
    numPoints = len(chosenPoints)
    if(interior == False):
        irradiationMatrix = get_irradiation_matrix(edgeList,obstacleNodeList,chosenPoints,obstacles,height=2,power=80,pseudo_3D=pseudo_3D,vsGraphList = vsGraphList)
    else:
        irradiationMatrix = get_irradiation_matrix(edgeList,obstacleNodeList,chosenPoints,obstacles.geoms[0].exterior,height=2,power=80,pseudo_3D=pseudo_3D)

    try:
        # Create a new model
        m = gp.Model("MinimizeDwellTime")
        # Creating variables
        # timeVarDict variables are named from 'C0' to 'CN', where N=numPoints-1
        timeVarDict = m.addMVar(numPoints, vtype=GRB.CONTINUOUS)
        slacks = m.addMVar(irradiationMatrix.shape[0], vtype = GRB.CONTINUOUS)
        # Setting objective
        print('before sum')
        m.setObjective(timeVarDict.sum() + 10000*slacks.sum(), GRB.MINIMIZE)
        print('after sum')
        # Setting constraints
            # Making sure all time variables are non-negative
        numPointZeros = np.zeros(numPoints)
        m.addConstr(timeVarDict >= numPointZeros, "Time_sign_constraints")
            # Adding constraints for required flux value for each edge
        minFluxVector = np.full(numEdges,minFluxReqd)
            # Multiply by -1 since the original irradiation matrix stores all negative values
        sparseIrradiationMatrix = sp.csr_matrix(-1 * irradiationMatrix)
        m.addConstr(sparseIrradiationMatrix @ timeVarDict + slacks >= minFluxVector, "Edge_flux_constraints")
        m.addConstr(slacks >= 0, "non-negative slacks")
        # Optimizing model
        m.optimize()
        # Getting solution
        if m.status == GRB.INFEASIBLE:
            print('Model is infeasible. Consider trying a different set of potential vantage points')
            return None, None, None
        # Otherwise, we have a solution and extract the values
        timeValues = list()
        # for v in m.getVars():
        timeValues = timeVarDict.x
            #print('%s %g' % (v.varName, v.x))
            # timeValues.append(v.x)

        print('\n\n\n slacks total = {} \n\n\n '.format(slacks.x.sum()))
        print('Obj: %g' % m.objVal)
        return timeValues, irradiationMatrix, m.objVal
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
    except AttributeError:
        print('Encountered an attribute error')

def compute_solid_angle(triangle,origin):
    R = triangle - origin
    orientation_matrix = np.ones(shape = (4,4))
    orientation_matrix[:3,:3] = R
    orientation_matrix[3,:3] = 0
    det = np.linalg.det(orientation_matrix)
    if(det > 0):
#         print('correct winding!')
        a = R[0,:]
        b = R[1,:]
        c = R[2,:]
    else:
#         print('wrong winding')
        a = R[0,:]
        b = R[2,:]
        c = R[1,:]
    det = np.linalg.det(R)
    norms = np.linalg.norm(R,axis = 1)
    a_norm = norms[0]
    b_norm = norms[1]
    c_norm = norms[2]
    nominator = np.dot(a,np.cross(b,c)) 
    denominator = a_norm*b_norm*c_norm + np.dot(a,b)*c_norm + np.dot(a,c)*b_norm + np.dot(b,c)*a_norm
    angle = 2*np.arctan2(nominator,denominator)
    if(angle < 0):
        angle = angle + np.pi
        
    return angle

def compute_flux_pseudo_3d(edge,point,height = 3.0,power = 1):
    triangle_1 = np.array([[edge[0][0],edge[0][1],-height/2.0],[edge[1][0],edge[1][1],-height/2.0],[edge[1][0],edge[1][1],height/2.0]])
    triangle_2 = np.array([[edge[1][0],edge[1][1],height/2.0],[edge[0][0],edge[0][1],height/2.0],[edge[0][0],edge[0][1],-height/2.0]])
    origin = np.array([point[0],point[1],0])
    solid_angle_1 = compute_solid_angle(triangle_1,origin)
    solid_angle_2 = compute_solid_angle(triangle_2,origin)
    edge_len = np.linalg.norm(np.array([edge[0][0],edge[0][1]])-np.array([edge[1][0],edge[1][1]]))
    
    return -(power*(solid_angle_1+solid_angle_2)/(4*np.pi))/(edge_len*height)

def find_vantage_points(filename, xRange, yRange, minDist):
    '''
    Takes a data file which contains the obstacle data for a room
    Obstacle data is of the form of a list of lists. Each nested list defines a single obstacle
    by defining the obstacle as a list of edges
    NOTE: Currently the edge list repeats the first edge twice, once at the beginning of the list
    and once at the end of the list
    Will return the vantage points at which the robot must stop, along
    with the times it must spend at each of these points
    '''
    with open(filename, 'rb') as f:
        polygonList = pickle.load(f).get_polygon_coordinates()

    edgeList = list()
    obstacleNodeList = list()
    for polygon in polygonList:
        # Here, assuming the polygon is defined by putting the same vertex at the beginning and end of the list
        for i in range(len(polygon)-1):
            edgeList.append((polygon[i], polygon[i+1]))
            obstacleNodeList.append(polygon[i])

    multiPolygonList = list()
    for val in polygonList:
        currPolygon = Polygon(val)
        multiPolygonList.append(currPolygon)
    
    obstacles = MultiPolygon(multiPolygonList)
    
    # Here we decide bounds and resolution depending on the min/max coordinates
    # How it is currently, this gives us a resolution of 1
    #gridPointList = get_grid_points(int(1*(xRange[1] - xRange[0] + 1)), int(1*(yRange[1] - yRange[0] + 1)), xRange, yRange, polygonList, minDist, float('inf'))
    gridPointList = get_grid_points_res(0.1, 0.1, xRange, yRange, obstacles, minDist)
    s2 = time.time()
    vsGraphs = get_vs_graphs(gridPointList,obstacleNodeList,edgeList,use_shapely=True,obstacles=obstacles)
    e2 = time.time()
    print('Just VS graphs time:', e2-s2)
    A_ub, solution = solve_lp(gridPointList, vsGraphs, edgeList, height=2, pseudo_3D=True)
    chosenPoints = list()
    for i in range(len(gridPointList)):
        currGridPtTime = solution.x[i]
        if currGridPtTime > 0:
            #print('Grid Point ', gridPointList[i], ' has time ', currGridPtTime)
            chosenPoints.append((gridPointList[i], currGridPtTime))
    return gridPointList, obstacleNodeList, edgeList, polygonList, chosenPoints, A_ub, solution

def get_dwell_times(polygonList, chosenPoints):

    multiPolygonList = list()
    for val in polygonList:
        currPolygon = Polygon(val)
        multiPolygonList.append(currPolygon)
    
    multiPolygon = MultiPolygon(multiPolygonList)
    #gridPointList = get_grid_points_res(0.25, 0.25,(0,4),(0,3),multiPolygon,0.1)
    edgeList = list()
    obstacleNodeList = list()
    for polygon in polygonList:
        # Here, assuming the polygon is defined by putting the same vertex at the beginning and end of the list
        for i in range(len(polygon)-1):
            edgeList.append((polygon[i], polygon[i+1]))
            obstacleNodeList.append(polygon[i])
    
    vsGraphs = get_vs_graphs(chosenPoints, obstacleNodeList, edgeList)
    A_ub, solution = solve_lp_gurobi(polygonList, chosenPoints, minFluxReqd, height = 2,power = 1,pseudo_3D = False)
    pointTimeList = list()
    for i in range(len(chosenPoints)):
        currTime = solution.x[i]
        if currTime > 0:
            pointTimeList.append((chosenPoints[i], currTime))

    return A_ub, solution, pointTimeList

def solve_transparent_edges(polygonList, checkGridPoints=False):
    edgeList = list()
    obstacleNodeList = list()
    for polygon in polygonList:
        # Here, assuming the polygon is defined by putting the same vertex at the beginning and end of the list
        for i in range(len(polygon)-1):
            edgeList.append((polygon[i], polygon[i+1]))
            obstacleNodeList.append(polygon[i])
    
    multiPolygonList = list()
    for val in polygonList:
        currPolygon = Polygon(val)
        multiPolygonList.append(currPolygon)
    multiPolygon = MultiPolygon(multiPolygonList)
    
    xMean = sum([x[0] for x in obstacleNodeList])/len(obstacleNodeList)
    yMean = sum([x[1] for x in obstacleNodeList])/len(obstacleNodeList)
    centroid = Point(xMean, yMean)
    numConstraints = len(edgeList)
    A_ub = np.zeros((numConstraints,1))

    for i in range(numConstraints):
        edgePoint1 = Point(edgeList[i][0])
        edgePoint2 = Point(edgeList[i][1])
        oppEdgeLen = edgePoint2.distance(edgePoint1)
        adjEdge1Len = centroid.distance(edgePoint1)
        adjEdge2Len = centroid.distance(edgePoint2)
        # Returns the angle subtended by the obstacle edge from the vantage point, using Law of Cosines
        angleSubtended = math.acos((adjEdge1Len**2 + adjEdge2Len**2 - oppEdgeLen**2)/(2*adjEdge1Len*adjEdge2Len))
        # The ratio of this angle to 2*pi gives the section of area of the circle which the edge receives radiation from
        A_ub[i][0] = -angleSubtended/(2*math.pi*oppEdgeLen)   
    
    b_ub = np.full((numConstraints,1), -1.0)
    b_ub -= 0.1 # Adding a buffer to make sure flux actually exceeds 1 for each edge
    c = np.ones((1, 1))
    
    # Having defined the LP, we can solve it
    sol = optimize.linprog(c, A_ub, b_ub, method="revised simplex") # By default, the variables must take non-negative real values, which is what we want
    
    if checkGridPoints == False:
        return centroid, A_ub, sol     
    
    currBestValue = sol.x[0]
    currBestPoint = centroid
    currBestSol = sol
    currBestIrradianceMatrix = np.copy(A_ub)
    gridPointList = get_grid_points_res(0.25, 0.25,(0,4),(0,3),multiPolygon,0.1)
    for point in gridPointList:
        currPoint = Point(point)
        for i in range(numConstraints):
            edgePoint1 = Point(edgeList[i][0])
            edgePoint2 = Point(edgeList[i][1])
            oppEdgeLen = edgePoint2.distance(edgePoint1)
            adjEdge1Len = currPoint.distance(edgePoint1)
            adjEdge2Len = currPoint.distance(edgePoint2)
            # Returns the angle subtended by the obstacle edge from the vantage point, using Law of Cosines
            angleSubtended = math.acos((adjEdge1Len**2 + adjEdge2Len**2 - oppEdgeLen**2)/(2*adjEdge1Len*adjEdge2Len))
            # The ratio of this angle to 2*pi gives the section of area of the circle which the edge receives radiation from
            A_ub[i][0] = -angleSubtended/(2*math.pi*oppEdgeLen)   
            if currPoint.x==2.5 and currPoint.y==0.5:
                print(angleSubtended, oppEdgeLen, " for value ", i)
        sol = optimize.linprog(c, A_ub, b_ub, method="revised simplex")
        if sol.x[0] < currBestValue:
            currBestValue = sol.x[0]
            currBestPoint = currPoint
            currBestSol = sol
            currBestIrradianceMatrix = np.copy(A_ub)
    return currBestPoint, currBestIrradianceMatrix, currBestSol

def get_midpoint_grid_points(polygonList, minDist):
    edgeList = list()
    obstacleNodeList = list()
    for polygon in polygonList:
        # Here, assuming the polygon is defined by putting the same vertex at the beginning and end of the list
        for i in range(len(polygon)-1):
            edgeList.append((polygon[i], polygon[i+1]))
            obstacleNodeList.append(polygon[i])

    multiPolygonList = list()
    for val in polygonList:
        currPolygon = Polygon(val)
        multiPolygonList.append(currPolygon)
    obstacles = MultiPolygon(multiPolygonList)

    chosenPoints = list()
    midpointEdges = list()
    for edge in edgeList:
        p1 = Point(edge[0])
        p2 = Point(edge[1])
        midPoint = Point((p1.x + p2.x)/2, (p1.y + p2.y)/2)
        currLine = LineString([p1,p2])
        leftLine = currLine.parallel_offset(1.05*minDist, 'left')
        rightLine = currLine.parallel_offset(1.05*minDist, 'right')
        potentialPoint1 = Point((leftLine.boundary[0].x + leftLine.boundary[1].x)/2, (leftLine.boundary[0].y + leftLine.boundary[1].y)/2)
        potentialPoint2 = Point((rightLine.boundary[0].x + rightLine.boundary[1].x)/2, (rightLine.boundary[0].y + rightLine.boundary[1].y)/2)

        # Checking these points are actually viable
        closestDistance = potentialPoint1.distance(obstacles)
        if(closestDistance < minDist):
            #print(' Too close!')
            pass
        else:
            #print('added Point 1')
            chosenPoints.append((potentialPoint1.x, potentialPoint1.y))
            midpointEdges.append(((potentialPoint1.x, potentialPoint1.y), (midPoint.x, midPoint.y)))

        closestDistance = potentialPoint2.distance(obstacles)
        if(closestDistance < minDist):
            #print('Too close')
            pass
        else:
            #print('added Point 2')
            chosenPoints.append((potentialPoint2.x, potentialPoint2.y))
            midpointEdges.append(((potentialPoint2.x, potentialPoint2.y), (midPoint.x, midPoint.y)))
    #print('no. of chosen points', len(chosenPoints))
    #print('no. of edges', len(edgeList))
    return chosenPoints, midpointEdges


def main():
    #gridPointList = get_grid_points(6, 6, (0,5), (0,5), examplePolygonList, 0.1, float('inf'))
    #gridPointList = get_grid_points(60, 42, (-14,15), (-6,14), examplePolygonList2, 0.2, float('inf')) 
    gridPointList = get_grid_points(24, 24, (0,11), (0,11), examplePolygonList3, 0.2, 0.5)
    startT = time.time()
    multiPolygonList = list()
    for val in examplePolygonList:
        currPolygon = Polygon(val)
        multiPolygonList.append(currPolygon)
    obstacles = MultiPolygon(multiPolygonList)
    vsGraphs = get_vs_graphs(gridPointList, exampleObstacleNodes3, exampleEdges3, use_shapely=True,obstacles=obstacles)
    A_ub, solution = solve_lp(gridPointList, vsGraphs, exampleEdges3)
    endT = time.time()
    print(solution.x)

    startT = time.time()
    # gridPointList, obstacleList, edgeList, polygonList, chosenPoints, A_ub, solution = find_vantage_points('pathFinding/adjacency_matrices/iteration_20/scene_20.p', (0,4), (0,4), 0.1)
    endT = time.time()
    print('done')

    timeSum = 0
    chosenPoints = list()
    for i in range(len(gridPointList)):
        currVal = solution.x[i]
        timeSum += currVal
        if currVal > 0:  # Can make the if statement 'if currVal > 0' when using 'simplex' or 'revised-simplex'
            print('Grid Point ', gridPointList[i], ' has time ', currVal)
            chosenPoints.append((gridPointList[i], currVal))
    print('No. of grid points: ', len(gridPointList))
    print('Solution time: ',timeSum)
    print('The number of chosen points is: ', len(chosenPoints))
    print('No. of edges is: ', len(edgeList))
    print('Time taken to attain solution is: ', endT - startT)
    #print(solution.x)
    # Plotting 
    myplot.plotGrid(gridPointList)
    #myplot.plotObstacles(exampleEdges3, 'blue')
    myplot.plotPointTimes(chosenPoints)
    myplot.plotEdgeFlux(edgeList3, A_ub, solution.x)
    myplot.show()

def get_internal_grid_points_res(x_res,y_res,xBounds,yBounds,obstacle,minDist):
    x_values = np.arange(xBounds[0],xBounds[1]+x_res,x_res)
    y_values = np.arange(yBounds[0],yBounds[1]+y_res,y_res)
    x_grid, y_grid = np.meshgrid(x_values,y_values)
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()
    gridPointList = []
    points = []
    for i in range(len(x_grid)):
        currx = x_grid[i]
        curry = y_grid[i]
        currPoint = Point(currx,curry)
        closestDistance = currPoint.distance(obstacle)
        if(closestDistance > 0):
            pass
        elif(obstacle.exterior.distance(currPoint) > minDist):
                gridPointList.append((currx,curry))
    return gridPointList

if __name__ == "__main__":
    main()