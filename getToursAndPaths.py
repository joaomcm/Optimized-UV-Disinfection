import math
import numpy as np
import time
import gridLP 
import plotSolutions as myplot
from dijkstar import Graph, find_path
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from klampt.plan import cspace,robotplanning
from klampt.io import resource
from klampt.plan.cspace import CSpace,MotionPlan
from klampt.vis.glprogram import GLProgram
from tspSolver5 import writeTSPLIBfile_FE, runTSP


costMatrix1 = np.array([[ 0.        ,  7.84259243, 12.23815657,  9.05400964],
       [ 7.84259243,  0.        , 13.27988077,  4.55543259],
       [12.23815657, 13.27988077,  0.        , 14.49129798],
       [ 9.05400964,  4.55543259, 14.49129798,  0.        ]])

examplePolygonList = [
[(4.5, 0.5),(4.5, 4.5),(2.5, 4.5),(2.5, 0.5)]
]
exampleObstacleNodes = [(4.5, 0.5),(4.5, 4.5),(2.5, 4.5),(2.5, 0.5)]
exampleEdges = [((4.5,0.5),(4.5,4.5)), ((4.5,4.5), (2.5, 4.5)), ((2.5, 4.5), (2.5, 0.5)), ((2.5,0.5), (4.5,0.5))]
exampleSolutionNodes = [(2.1739130434782608, 2.391304347826087), (3.4782608695652173, 0.21739130434782608), (3.4782608695652173, 4.782608695652174), (4.782608695652174, 2.391304347826087)]

class RobotCSpace(CSpace):
    def __init__(self, xRange, yRange, minDistance):
        CSpace.__init__(self)
        #set bounds
        self.bound = [(xRange[0], xRange[1]), (yRange[0], yRange[1])]
        #set collision checking resolution
        self.eps = 1e-3 
        #set obstacles here
        self.obstacles = []
        self.minDistance = minDistance # Sets the distance which a robot needs to exceed from any obstacle
    
    def addObstacle(self, polygon):
        self.obstacles.append(polygon)

    def feasible(self,q):
        #bounds test
        qPoint = Point(q)
        # Checks if a point is too close to an obstacle for all obstacles
        if qPoint.x > self.bound[0][1] or qPoint.x < self.bound[0][0] or qPoint.y > self.bound[1][1] or qPoint.y < self.bound[1][0]:
            return False
            
        isFeasible = True
        for polygon in self.obstacles:
            # currPolygon = Polygon(polygon)
            closestDistance = qPoint.distance(polygon)
            if closestDistance <= self.minDistance:
                isFeasible = False
                break
        return isFeasible

class CSpaceObstacleProgram(GLProgram):
    def __init__(self,space, start, goal, initial_points = 1000):
        GLProgram.__init__(self)
        self.space = space

        #PRM planner
        MotionPlan.setOptions(type="prm*",knn=10,connectionThreshold=0.2) # Change type based on what planner you want to use
        self.optimizingPlanner = True

        self.planner = MotionPlan(space)
        self.start = start
        self.goal = goal
        self.planner.addMilestone(start)
        self.planner.addMilestone(goal)
        self.components = int(self.planner.getStats()['numComponents'])
        print(self.components)
        self.path = []
        self.G = None

def getSinglePath(startNode, endNode, xBounds, yBounds, obstacleList, minDistance):
    '''
        Takes two milestone vertices from the graph and finds a feasible (close-to-optimal?) path from one to the other
    '''
    space = RobotCSpace(xBounds, yBounds, minDistance)
    for obstacle in obstacleList:
        polygon = Polygon(obstacle)
        space.addObstacle(polygon)
    planner = CSpaceObstacleProgram(space, start = startNode, goal = endNode)
    increment = 500 # Change this if need be?
    t0 = time.time()
    while True:   #max 10 seconds of planning
        planner.planner.planMore(increment)
        path = planner.planner.getPath()
        if path and len(path) > 0:
            print("Solved, path has ",len(path)," milestones")
            print("Took time ",time.time()-t0)
            break
    V,E = planner.planner.getRoadmap()
    planner.planner.close()
    return V, E, path

def getTour(costMatrix, filename, user_comment='This is a tour for the given problem'):
    '''
        Takes a distance matrix (not necessarily Euclidean distances) for a set of points, and uses a TSP
        solver to find a good tour among all of them. The tour is written to a text file, specified by filename
        along with a user comment, if so desired 
    '''

    runTSP(costMatrix, filename, user_comment)

def readTourFile(filename, milestones):
    '''
        Reads the file which had the tour written to it and extracts the tour
    '''
    
    addLine = False
    getFirstNode = False
    firstNode = None
    tour = list()
    pathLength = 0
    fp = open(filename, 'r')
    lines = fp.read().splitlines()
    startTourIndex = lines.index('TOUR_SECTION')
    endTourIndex = lines.index('-1')
    stringTour = lines[startTourIndex + 1: endTourIndex]
    integerTour = [(int(val) - 1) for val in stringTour] # In the LKH solver, the nodes are indexed starting from 1, so we reduce 1 to start indexing from 0
    # tour.append(tour[0])
    # Now, we convert the integer values to their corresponding milestones in Euclidean coordinates
    # euclideanTour = [milestones[integerTour[i]] for i in range(len(integerTour))]

    return integerTour

def getFinalPath(tspTour, xBounds, yBounds, obstacleList, minDistance):
    '''
        Takes a given tour of a set of points in space, and returns the overall path that must be taken to reach all of them
        The final path will described actual vertices as Euclidean coordinates.
    '''

    # In the given 
    overallPath = list()
    for i in range(len(tspTour) - 1):
        V,E,path = getSinglePath(tspTour[i], tspTour[i+1], xBounds, yBounds, obstacleList, minDistance)
        overallPath += path[:-1] # Not adding the last node on the path because it will be repeated on the next path
    overallPath.append(tspTour[-1])
    return overallPath

def getPathFromPrm(tspTour,pathDict,milestones, polygonList, minDist):
    '''
    '''
    multiPolygonList = list()
    for val in polygonList:
        currPolygon = Polygon(val)
        multiPolygonList.append(currPolygon)
    obstacles = MultiPolygon(multiPolygonList)
    overallPath = list()
    for i in range(len(tspTour) - 1):
        directEdge = LineString([milestones[tspTour[i]], milestones[tspTour[i+1]]])
        distance = directEdge.distance(obstacles)
        if distance > minDist:
            overallPath.append(milestones[tspTour[i]])
            overallPath.append(milestones[tspTour[i+1]])
        else:
            currEdge = (tspTour[i], tspTour[i+1])
            if currEdge[0] < currEdge[1]:
                currPath = pathDict[currEdge].copy()
            else:
                currPath = pathDict[(currEdge[1],currEdge[0])].copy()
                currPath.reverse()
            overallPath += currPath[:-1]
    overallPath.append(milestones[tspTour[-1]])
    return overallPath

def getFinalPath(nodeTour,pathDict,gridPointList,chosenPointIndices):
    '''
        Function intended for use with MILP for minimizing total time. It uses a grid graph to generate paths
        between pairs of points, and given a tour of some subset of these points, returns a feasible path through
        the structure. Here, both nodeTour and pathDict use indices of values in gridPointList to describe those points
    '''
    pathMilestones = list()
    pathEdges = list()
    pathLength = 0
    for i in range(len(nodeTour)-1):
        currEdge = (nodeTour[i],nodeTour[i+1])
        currPath = pathDict[chosenPointIndices[currEdge[0]]][chosenPointIndices[currEdge[1]]]
        pathWithNodes = [gridPointList[val] for val in currPath] 
        pathMilestones += pathWithNodes[:-1]
    pathMilestones.append(gridPointList[chosenPointIndices[nodeTour[-1]]])
    for i in range(len(pathMilestones)-1):
        pathEdges.append((pathMilestones[i], pathMilestones[i+1]))
        point1 = Point(pathMilestones[i])
        point2 = Point(pathMilestones[i+1])
        pathLength += point1.distance(point2) # This is correct, because all pairs of edges are between straight lines on the grid - hence Euclidean distances
    return pathMilestones, pathEdges, pathLength

def main():
    # getTour(costMatrix1, 'currTSP')
    # tour = readTourFile('currTSP.txt')
    # print(tour)

    V, E, path = getSinglePath((0.5,0.5), (1.5, 3), (0,5), (0,5), examplePolygonList, 0.2)
    myplot.plotChosenPoints(V)
    myplot.plotObstacles(exampleEdges, "blue")
    pathEdges = [(V[e[0]], V[e[1]]) for e in E]
    myplot.plotObstacles(pathEdges, "green")
    actualPath = list()
    for i in range(len(path)-1):
        actualPath.append((path[i], path[i+1]))
    myplot.plotObstacles(actualPath, "red")
    myplot.show()

if __name__ == "__main__":
    main()
