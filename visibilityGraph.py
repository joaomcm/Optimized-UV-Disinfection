import math
import numpy
from shapely.geometry import Point, LineString
import shapely

shape = [[(-6,1)],
         [(-13,2),(-13,1),(-12,1),(-12,2)],
         [(-11,4),(-11,-2),(-9,-2),(-9,4)],
         [(-8,7),(-7,6),(-6,6),(-5,7),(-6,8),(-7,8)],
         [(-2,8),(-2,0),(0,0),(0,8)],
         [(3,6),(4,6),(5,7),(4,8),(3,8),(2,7)],
         [(11,12),(11,7),(13,7),(13,12)],
         [(13,5),(13,2),(12,2),(12,5)],
         [(13,-3),(11,-5),(14,-5)]
        ]
exampleStartNode = shape[0][0]
exampleObstacleNodes = [(-13,2),(-13,1),(-12,1),(-12,2),
                        (-11,4),(-11,-2),(-9,-2),(-9,4),
                        (-8,7),(-7,6),(-6,6),(-5,7),(-6,8),(-7,8),
                        (-2,8),(-2,0),(0,0),(0,8),
                        (3,6),(4,6),(5,7),(4,8),(3,8),(2,7),
                        (11,12),(11,7),(13,7),(13,12),
                        (13,5),(13,2),(12,2),(12,5),
                        (13,-3),(11,-5),(14,-5)]
exampleEdges = [((-13,2),(-13,1)), ((-13,1),(-12,1)),((-12,1),(-12,2)),((-12,2),(-13,2)),
                ((-11,4),(-11,-2)),((-11,-2),(-9,-2)),((-9,-2),(-9,4)),((-9,4),(-11,4)),
                ((-8,7),(-7,6)),((-7,6),(-6,6)),((-6,6),(-5,7)),((-5,7),(-6,8)),((-6,8),(-7,8)),((-7,8),(-8,7)),
                ((-2,8),(-2,0)),((-2,0),(0,0)),((0,0),(0,8)),((0,8),(-2,8)),
                ((3,6),(4,6)),((4,6),(5,7)),((5,7),(4,8)),((4,8),(3,8)),((3,8),(2,7)),((2,7),(3,6)),
                ((11,12),(11,7)),((11,7),(13,7)),((13,7),(13,12)),((13,12),(11,12)),
                ((13,5),(13,2)),((13,2),(12,2)),((12,2),(12,5)),((12,5),(13,5)),
                ((13,-3),(11,-5)),((11,-5),(14,-5)),((14,-5),(13,-3))]

exampleStartNode2 = (0,0)
exampleObstacleNodes2 = [(5,1),(5,5),(3,5),(3,1)]
exampleEdges2 = [((5,5),(3,5)),((3,5),(3,1)),((3,1),(5,1)),((5,1),(5,5))]

def sort_nodes(startNode, nodeList):
    '''
        Will sort all the nodes in nodeList in increasing order of rotational angle with respect to
        startNode
        Ties are broken by how close a node is to startNode
    '''
    startX = startNode[0]
    startY = startNode[1]
    nodeInfoList = list()
    for node in nodeList:
        xCoord = node[0] - startX
        yCoord = node[1] - startY
        angle = math.atan2(yCoord,xCoord)
        if(angle<0):
            angle = angle + 2*math.pi
        euclideanDist = math.sqrt(xCoord**2 + yCoord**2)
        nodeInfoList.append((node, angle, euclideanDist))
    sortedNodeInfoList = sorted(nodeInfoList, key = lambda element: (element[1], element[2]))
    sortedNodeList = [node[0] for node in sortedNodeInfoList]

    return sortedNodeList

def get_distance(ray1, ray2, segment1, segment2):
    '''
    Input:
        ray1 = (x1, y1) - the point at which the ray originates
        ray2 = (x2, y2)
        segment1 = (x3, y3)
        segment2 = (x4, y4)
    Output:
        Returns the distance between the ray and the segment if they intersect; else, returns float('inf')
    '''
    startPoint = Point(ray1)
    # First, we generate a segment from the points ray1 and ray2 which is long enough, such that
    # if the segment and infinite ray intersect, then so will this segment of the ray and the other line segment
    xIncrease = ray2[0] - ray1[0]
    yIncrease = ray2[1] - ray1[1]
    if xIncrease > 0:
        # If the x-coordinate increases in the ray's direction
        xMaxValue = max(segment1[0], segment2[0])
        c = (xMaxValue - ray1[0])/xIncrease + 1
    elif xIncrease < 0:
        # If the x-coordinate decreases in the ray's direction
        xMinValue = min(segment1[0], segment2[0])
        c = (xMinValue - ray1[0])/xIncrease + 1
    # If we reach here, the ray has slope infinity (or negative infinity), so yIncrease != 0
    elif yIncrease > 0:
        yMaxValue =  max(segment1[1], segment2[1])
        c = (yMaxValue - ray1[1])/yIncrease + 1
    elif yIncrease < 0:
        yMinValue = min(segment1[1], segment2[1]) 
        c = (yMinValue - ray1[1])/yIncrease + 1
    newRay2 = (ray1[0] + c*xIncrease, ray1[1] + c*yIncrease)

    # Find the intersection point of the "ray" and the segment 
    ray = LineString([ray1, newRay2])
    segment = LineString([segment1, segment2])
    intersectionPoint = ray.intersection(segment)
    
    # Checking that the intersection point actually exists
    if isinstance(intersectionPoint, Point):
        distance = startPoint.distance(intersectionPoint)
        return distance
    else:
        # Otherwise, they didn't intersect
        return float('inf')

def get_distance2(ray1, ray2, segment1, segment2):
    '''
    Input:
        ray1 = (x1, y1) - the point at which the ray originates
        ray2 = (x2, y2)
        segment1 = (x3, y3)
        segment2 = (x4, y4)
    Output:
        Returns the distance between the ray and the segment if they intersect; else, returns
        the distance between 
    '''
    startPoint = Point(ray1)
    # First, we generate a segment from the points ray1 and ray2 which is long enough, such that
    # if the segment and infinite ray intersect, then so will this extended ray segment and the segment
    xIncrease = ray2[0] - ray1[0]
    yIncrease = ray2[1] - ray1[1]
    if xIncrease > 0:
        # If the x-coordinate increases in the ray's direction
        xMaxValue = max(segment1[0], segment2[0])
        c = (xMaxValue - ray1[0])/xIncrease + 1
    elif xIncrease < 0:
        # If the x-coordinate decreases in the ray's direction
        xMinValue = min(segment1[0], segment2[0])
        c = (xMinValue - ray1[0])/xIncrease + 1
    # If we reach here, the ray has slope infinity (or negative infinity), so yIncrease != 0
    elif yIncrease > 0:
        yMaxValue =  max(segment1[1], segment2[1])
        c = (yMaxValue - ray1[1])/yIncrease + 1
    elif yIncrease < 0:
        yMinValue = min(segment1[1], segment2[1]) 
        c = (yMinValue - ray1[1])/yIncrease + 1
    newRay2 = (ray1[0] + c*xIncrease, ray1[1] + c*yIncrease)

    # Find the intersection point of the two lines generated by these points
    ray = LineString([ray1, newRay2])
    segment = LineString([segment1, segment2])
    intersectionPoint = ray.intersection(segment)
    
    # Checking that the intersection point actually exists
    if isinstance(intersectionPoint, Point):
        distance = startPoint.distance(intersectionPoint)
        return distance
    else:
        currEdgeStartPoint = Point(segment1)
        return currEdgeStartPoint.distance(startPoint)

def get_visibility_edges(startNode, totalEdgeList, sortedNodeList):
    '''
    Given a starting node and a list of obstacle nodes (sorted by increasing angle) along with the edges they create,
    finds the vertices v in the node list such that the edge from startNode to v is entirely
    in open space, and returns these edges, i.e. finds all visible vertices from the given starting node

    Uses the rotational sweep algorithm
    '''

    intersectingEdgeList = list()
    minAngleNode = sortedNodeList[0]
    for edge in totalEdgeList:
        dist = get_distance(startNode, minAngleNode, edge[0], edge[1])
        if dist != float('inf'):
            intersectingEdgeList.append((edge, dist))
    # Now, sort this edge list (all collision edges) by distance from startNode
    sortedEdgeList = sorted(intersectingEdgeList, key = lambda element: element[1])
    visibleEdges = list()
    print('s',sortedEdgeList)
    if minAngleNode in sortedEdgeList[0][0]:
        visibleEdges.append((startNode, minAngleNode))
    # Now, go through all nodes except the first one
    for i in range(1, len(sortedNodeList)):
        w = sortedNodeList[i]
        if w == sortedEdgeList[0][0][1]:
            # If w is the end of sortedEdgeList[0], then we remove that edge 
            # from I and add (startNode, w) to E (list of visible edges)
            visibleEdges.append((startNode, w))
            sortedEdgeList.remove(sortedEdgeList[0])
            print(visibleEdges)
        else:
            # Check if w is the end vertex of any other edge in the sortedEdgeList and filter out
            # those edges
            sortedEdgeList = [edge for edge in sortedEdgeList if w != edge[0][1]] # Recall sortedEdgeList is a list of tuples (edge, dist)
            wStartingEdges = [edge for edge in totalEdgeList if w == edge[0][0]]
            print('www',wStartingEdges)
            print('sss',sortedEdgeList)
            for edge in wStartingEdges:
                justEdges = [edge[0] for edge in sortedEdgeList]
                if edge not in justEdges:
                    # I'm using get_distance2 here to implement the part we talked about on Slack
                    # It's mostly the same code as get_distance except for the final return
                    edgeTuple = (edge, get_distance2(startNode, minAngleNode, edge[0], edge[1]))
                    index = 0
                    while index < len(sortedEdgeList) and sortedEdgeList[index][1] < edgeTuple[1]:
                        index += 1
                    sortedEdgeList.insert(index, edgeTuple)
    return visibleEdges

def generate_visibility_graph(startNode, nodeList, edgeList):
    '''
        Given a start node, a list of obstacle vertices and the corresponding edges,
        returns a graph of edges that are traversable (i.e. the visibility graph)
    '''
    sortedNodeList = sort_nodes(startNode, nodeList)
    print(sortedNodeList)
    visibleEdges = get_visibility_edges(startNode, edgeList, sortedNodeList)
    return visibleEdges

def generate_visibility_graph_no_rotation(startNode, nodeList, edgeList,use_shapely = False,obstacles = []):
    '''
    Finds the edges which are traversable
    '''
    if(not use_shapely):
        visibleEdges = list()
        startPoint = Point(startNode)
        for node in nodeList:
            nodePoint = Point(node)
            line1 = LineString([startPoint, nodePoint])
            flag = 0
            for edge in edgeList:
                if edge[0] != node and edge[1] != node:
                    line2 = LineString([edge[0], edge[1]])
                    intersectionPoint = line1.intersection(line2)
                    # Checking that the intersection point actually exists
                    if isinstance(intersectionPoint, Point):
                        flag = 1
                        break
            if flag == 0:
                visibleEdges.append((startNode, node))
        return visibleEdges
    else:
        visibleEdges = list()
        startPoint = Point(startNode)
        for node in nodeList:
            nodePoint = Point(node)
            line = LineString([startPoint, nodePoint])
        #     flag = 0
        #     for edge in edgeList:
        #         if edge[0] != node and edge[1] != node:
        #             line2 = LineString([edge[0], edge[1]])
        #             intersectionPoint = line1.intersection(line2)
        #             # Checking that the intersection point actually exists
        #             if isinstance(intersectionPoint, Point):
        #                 flag = 1
        #                 break
        #     if flag == 0:
            inter = obstacles.intersection(line)
            if(type(inter) ==  shapely.geometry.point.Point):
                # display(GeometryCollection([mp,line]))
                visibleEdges.append((startNode, node))
        return visibleEdges



def main():
    visibleEdges = generate_visibility_graph_no_rotation(exampleStartNode, exampleObstacleNodes, exampleEdges)
    print(visibleEdges)

if __name__ == "__main__":
    main()
