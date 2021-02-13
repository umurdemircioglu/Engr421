import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
means = [ [0, 2.5], [-2.5, -2.0], [2.5, -2.0] ]
sigmas = [ [[3.2, 0], [0, 1.2 ]], [[1.2, -0.8], [-0.8, 1.2]], [[1.2, 0.8], [0.8, 1.2]] ]
n = [120, 90, 90]
colors = ['r.','g.','b.']
sample_means = [None, None, None]
cov = [None, None, None]
priors = [None, None, None]
points = [[],[]]

for i in range(3):
    p = np.random.multivariate_normal(means[i], sigmas[i],n[i])
    x1,x2 = p.T
    points[0] += x1.tolist()
    points[1] += x2.tolist()
    plt.plot(x1, x2, colors[i])
    sample_means[i] = [sum(x1)/len(x1), sum(x2)/len(x2)]
    tmp = np.array([ x1 - sample_means[i][0], x2 - sample_means[i][1]])
    cov[i] = (np.matmul(tmp, tmp.T )) / n[i]
    priors[i] = [n[i]/sum(n)]

X = np.array(points)
y = np.concatenate((np.repeat(1, n[0]), np.repeat(2, n[1]), np.repeat(3, n[2])))

def score(x,v,m,p):
       return (-np.log(2*math.pi)) -1/2*(np.log(np.linalg.norm(v))) -1/2*np.matmul((x-m).T,np.matmul((np.linalg.inv(v)),(x-m)))+np.log(p)

ypred = []
for j in range(len(X[0])):
    idx = 0
    sc_max = -9999999999999999999
    for i in range(3):
        sc = score(X[:, j], cov[i], sample_means[i],priors[i])
        if sc > sc_max:
            idx = i
            sc_max = sc
    ypred.append(idx)


ypred = np.array(ypred) + 1
confusion_matrix = pd.crosstab(ypred, y, rownames = ['ypred'], colnames = ['y'])

print('Means')
print(sample_means)
print('Covariances')
print(cov)
print('Class Priors')
print(priors)
print('Confusion Matrix')
print(confusion_matrix)

n = 200
x1_interval = np.linspace(-6, +6, n)
x2_interval = np.linspace(-6, +6, n)
xx, yy = np.meshgrid(x1_interval, x2_interval)
Z = np.zeros((n, n))

for k in range(n):
    X = np.vstack((xx[:,k], yy[:,k]))
    for j in range(len(X[0])):
        idx = 0
        sc_max = -9999999999999999999
        for i in range(3):
            sc = score(X[:, j], cov[i], sample_means[i],priors[i])
            if sc > sc_max:
                idx = i
                sc_max = sc
        Z[k,j] = idx
plt.contour(yy,xx, Z, colors='black')
plt.contourf(yy,xx, Z)
plt.plot(np.array(points).T[ypred != y, 0], np.array(points).T[ypred != y, 1], "ko", markersize = 12, fillstyle = "none")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
