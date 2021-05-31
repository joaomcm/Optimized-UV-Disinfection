import math
import numpy as np
import time 
import .plotSolutions as myplot
from dijkstar import Graph, find_path
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from klampt.plan import cspace,robotplanning
from klampt.io import resource
from klampt.plan.cspace import CSpace,MotionPlan
from klampt.vis.glprogram import GLProgram
from planning.tsp_solver_wrapper import writeTSPLIBfile_FE, runTSP






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

    exampleObstacleNodes = [(4.5, 0.5),(4.5, 4.5),(2.5, 4.5),(2.5, 0.5)]
    exampleEdges = [((4.5,0.5),(4.5,4.5)), ((4.5,4.5), (2.5, 4.5)), ((2.5, 4.5), (2.5, 0.5)), ((2.5,0.5), (4.5,0.5))]

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
