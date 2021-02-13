import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


datatrain = np.zeros((1,2))
datatest = np.zeros((1,2))


data = pd.read_csv("hw04_data_set.csv",header = None).to_numpy()

datatrain = np.vstack([datatrain, data[1:101, :]])
datatest = np.vstack([datatest, data[101:134, :]])

datatrain = np.delete(datatrain, 0, 0)
datatest = np.delete(datatest, 0, 0)
datatrain = datatrain.astype(np.float)
datatest = datatest.astype(np.float)

xhigh = np.max(datatrain[:,0])
xlow = np.min(datatrain[:,0])
binwidth = 3
origin = 0
left_borders = np.arange(origin, xhigh, binwidth)
right_borders = np.arange(origin + binwidth, xhigh + binwidth, binwidth)

def regressogram(lborder,rborder):
    top = 0
    bottom = 0
    for i in range(100):
        if(lborder < datatrain[i,0] <= rborder):
            top = top + datatrain[i,1]
            bottom = bottom + 1
    return top/bottom

g = []
for i in range(20):
    g.append(regressogram(left_borders[i],right_borders[i]))
g = np.asarray(g)
plt.plot(datatest[:,0],datatest[:,1], "r.", markersize = 10)
plt.plot(datatrain[:,0],datatrain[:,1], "b.", markersize = 10)
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [g[b], g[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [g[b], g[b + 1]], "k-")
plt.show()

def RMSE(gg,dtest):
    tmp = 0
    for i in range(20):
        for j in range(33):
            if(left_borders[i]< datatest[j,0] <= right_borders[i]):
                a = (datatest[j,1]-gg[i])**2
                tmp = tmp + a
    return np.sqrt(tmp/len(dtest))
rmse1 = RMSE(g,datatest)
print("Regressogram => RMSE is",rmse1,"when h is 3" )

def RunningMeanSmoother(x):
    top = 0
    bottom = 0
    for i in range(100):
        if(np.abs((x - datatrain[i,0])/3) < 1):
            top = top + datatrain[i,1]
            bottom = bottom + 1
    return top/bottom

def RMSE2(gg,dtest,x):
    tmp = 0
    for x in range(1601):
        for j in range(33):
            if(np.abs((x-datatest[i,0])/3) < 1):
                a = (datatest[j,1]-gg[x])**2
                tmp = tmp + a
    return np.sqrt(tmp/len(dtest))

g1 = []
data_interval = np.linspace(xlow, xhigh, 1601)
for x in range(1601):
    g1.append(RunningMeanSmoother(data_interval[x]))
g1 = np.asarray(g1)
plt.plot(datatest[:,0],datatest[:,1], "r.", markersize = 10)
plt.plot(datatrain[:,0],datatrain[:,1], "b.", markersize = 10)
plt.plot(data_interval, g1, "k-")
plt.show()

rmse2 = RMSE2(g1,datatest,data_interval[x])
print("RunningMeanSmoother => RMSE is",rmse2,"when h is 3" )
