import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.rcParams.update({'font.size': 10,'font.weight':'bold'})


def plotGrid(gridPoints):
    plt.clf()
    #fig = plt.figure(figsize = (15,15))
    x = list()
    y = list()
    for i in gridPoints:
        x.append(i[0])
        y.append(i[1])
    plt.scatter(x,y, c = 'blue', alpha = 0.6, s = 1)

def plotGridLines(edgeList, color):
    for edge in edgeList:
        x1 = (edge[0][0], edge[1][0])
        x2 = (edge[0][1], edge[1][1])
        plt.plot(x1,x2,color=color,linewidth=0.8)

def plotObstacles(edgeList, color):
    segs = list()
    for edge in edgeList:
        x1 = (edge[0][0], edge[1][0])
        x2 = (edge[0][1], edge[1][1])
        segs.append(x1)
        segs.append(x2)
        segs.append(color)
    plt.plot(*segs)

def plotChosenPoints(chosenPoints, label = False):
    x = list()
    y = list()
    for i in chosenPoints:
        x.append(i[0])
        y.append(i[1])
    n = list()
    for i in range(len(chosenPoints)):
        n.append(i)
    plt.scatter(x,y, c = 'red', alpha = 0.9, s = 12.5)
    if label == True:
        for i, txt in enumerate(n):
            plt.annotate(txt, (x[i], y[i]))

def plotPointTimes(chosenPoints):
    '''
        Input: chosenPoints: a list of tuples, each of which consists of a point along with the time 
                             that must be spent at that point
        Output: None. Plots the points, with different colorings based on the amount of time that must be spent 
                at that point. Currently, we do:
                Green: 1 - 5 seconds
                Orange: 5 - 30 seconds
                Red: > 30 seconds
    '''
    x = list()
    y = list()
    c = list()
    minTime = float('inf') # Time has to be non-negative
    maxTime = 0
    for (i,t) in chosenPoints:
        x.append(i[0])
        y.append(i[1])
        c.append(t)
        if t < minTime:
            minTime = t
        if t > maxTime:
            maxTime = t
    df = pd.DataFrame({"x":x, "y":y, "c": c})
    cmap = plt.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=minTime, vmax=maxTime)
    #fig, ax = plt.subplots()
    plt.scatter(df.x, df.y, color = cmap(norm(df.c.values)), s = 16)
    sm = plt.cm.ScalarMappable(cmap = cmap, norm = norm)
    sm.set_array([])
    cbar1 = plt.colorbar(sm)
    cbar1.set_label("Dwell-time of points (in seconds)")


def plotEdgeFlux(edgeList, irradiationMatrix, solution,cap_fluence = False,cap = 3*280):
    '''
        Input: edgeList: a list of tuples, each of which consists of an edge along with the amount of flux
                             received by that edge given a solution of points along with their times
        Output: None. Plots the obstacle edges, with different colorings based on the amount of flux that  
                each edge receives. Currently, we do:
                Green: 1 - 2
                Orange: 2 - 4
                Red: > 4
    '''
    edgeFluxValues = -1 * np.dot(irradiationMatrix, solution) # Recall that the constraints all take negative values
    minFlux = min(edgeFluxValues)
    maxFlux = max(edgeFluxValues)
    print(minFlux, maxFlux)
    cmap = plt.cm.viridis
    if(cap_fluence == False):
        norm = matplotlib.colors.Normalize(vmin=min(minFlux,0), vmax=450)
    else:
        norm = matplotlib.colors.Normalize(vmin=min(minFlux,0), vmax=450)
    df = pd.DataFrame({"c": edgeFluxValues})
    colorList = cmap(norm(df.c.values))
    segs = list()
    for i in range(len(edgeList)):
        edge = edgeList[i]
        fluxVal = edgeFluxValues[i]
        # print('Edge ', i, ' ', edge, ' with ', irradiationMatrix[i], ' and flux value ', fluxVal)
        x1 = (edge[0][0], edge[1][0])
        x2 = (edge[0][1], edge[1][1])
        segs.append(x1)
        segs.append(x2)
        segs.append(colorList[i])
        plt.plot(x1, x2, c = colorList[i])
    sm = plt.cm.ScalarMappable(cmap = cmap, norm = norm)
    sm.set_array([])
    cbar2 = plt.colorbar(sm)#, ticks=[minFlux, (minFlux+maxFlux)/2, maxFlux])
    cbar2.set_label("UV fluence over the edges (J/m^2)",fontsize = 12,weight = 'bold')
    plt.xlabel('room length (m)',fontsize = 12,weight = 'bold')
    plt.ylabel('room width (m)',fontsize = 12,weight = 'bold')

def show():
    plt.show()

def clear():
    plt.clf()

def savefig(figname = 'currImage.pdf'):
    plt.savefig(figname,bbox_inches = 'tight', format = 'pdf')
    plt.clf()

def main():
    dwellOptimal = [280.2842394,462.8895857,493.9026778,447.4919439,348.8299663,336.8899703,451.0602088,479.2481303,227.6317933,425.3969176]
    disinfectionOptimal = [306.6842394,492.4895857,529.1026778,475.4919439,373.6299663,364.0899703,485.4602088,508.8481303,248.4317933,456.5969176]
    pathOptimal = [16.5,18.5,22,17.5,15.5,17,21.5,18.5,13,19.5]
    numPointsOptimal = [25,21,27,20,16,23,33,27,18,22]
    runOptimal = [111.0781636,42.8900013,54.8610034,54.44558525,48.39299345,55.07403517,226.1797941,78.12079191,84.91803718,27.75956631]

    dwellApprox = [280.0816362,462.8895857,493.9026778,446.9341249,348.8299663,336.8899703,449.6751471,477.569803,227.6317933,415.4674643]
    disinfectionApprox = [322.0816362,502.8895857,543.9026778,485.9341249,381.8299663,372.8899703,497.6751471,519.569803,255.6317933,461.4674643]
    pathApprox = [21,20,25,19.5,16.5,18,24,21,14,23]
    numPointsApprox = [25,21,27,21,16,23,34,29,18,28]
    runApprox = [2.169002533,2.250688314,3.443999529,2.45172596,2.079940796,2.061151266,3.267003298,2.322207689,1.866064787,2.802033901]

    percentage = [0.130401722, 0.086082024,0.079540323,0.060102794,0.091928211,0.06652773,0.080257792,0.058886381
    ,0.086425904,0.066375832,0.096543224,0.074706809,0.096448457,0.070860597,0.080836107,0.058170598,0.109532541,0.083725194,
    0.114671768,0.093737301,0.068331605]

    x = [100*i for i in percentage]

    histList = list()
    n_bins = 20
    for i in range(10):
        val = (runOptimal[i]-runApprox[i])/runApprox[i]
        histList.append(val)
    plt.title('Run-time (LP + TSP')
    plt.xlabel('Traversal Time / Total Disinfection Time')
    plt.ylabel('No. of examples')
    plt.hist(histList,bins=n_bins)
    plt.xlim((0,1))
    plt.show()
    # x = [0.05,0.1,0.25,0.5,1]
    # y = [357.83142733573914,52.46800184249878,24.661519050598145,16.73600149154663,13.701037406921387]
    # n = ['0.1', '0.25', '0.5', 'Resolution = 0.75']
    # for i, txt in enumerate(n):
    #     plt.annotate(txt, (x[i], y[i]))
    # plt.clr()
    # x2 = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]#, 0.5]
    # y2 = [1, 1.043438215,1.164422775,1.337089702,1.784411148,4.780248255]#,60.78718235]
    # plt.scatter(x2,y2,c = "blue")
    # plt.plot(x2,y2,c="blue")
    plt.show()


if __name__ == "__main__":
    main()

