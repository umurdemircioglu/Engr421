import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def safelog(x):
        return(np.log(x + 1e-100))

labelstrain = []
labelstest = []
imagestrain = np.zeros((1,320))
imagestest = np.zeros((1,320))
meanstrain = np.zeros((1,320))
meanstest = np.zeros((1,320))


labels = pd.read_csv("hw03_data_set_labels.csv",header = None).to_numpy()
images = pd.read_csv("hw03_data_set_images.csv",header = None).to_numpy()

ytruth = np.concatenate((np.repeat(1, 25), np.repeat(2,25), np.repeat(3, 25),np.repeat(4, 25),np.repeat(5, 25)))
ytruthtest = np.concatenate((np.repeat(1, 14), np.repeat(2,14), np.repeat(3, 14),np.repeat(4, 14),np.repeat(5, 14)))


for i in range(5):
    labelstrain = np.append(labelstrain,labels[i*39:i*39+25, :])
    labelstest = np.append(labelstest, labels[i*39+25:i*39+39, :])
    imagestrain = np.vstack([imagestrain, images[i*39:i*39+25, :]])
    imagestest = np.vstack([imagestest, images[i*39+25:i*39+39, :]])
    meanstrain = np.vstack([meanstrain, images[i*39:i*39+25, :].mean(axis=0).transpose()])
    meanstest = np.vstack([meanstest, images[i*39+25:i*39+39, :].mean(axis=0).transpose()])


imagestrain = np.delete(imagestrain, 0, 0)
imagestest= np.delete(imagestest, 0, 0)
meanstrain = np.delete(meanstrain, 0, 0)
meanstest= np.delete(meanstest, 0, 0)

classpriors = [1/5,1/5,1/5,1/5,1/5]

pcd1 = meanstrain[0]
pcd2 = meanstrain[1]
pcd3 = meanstrain[2]
pcd4 = meanstrain[3]
pcd5 = meanstrain[4]

print(" pcd1 = ", pcd1)
print(" pcd2 = ", pcd2)
print(" pcd3 = ", pcd3)
print(" pcd4 = ", pcd4)
print(" pcd5 = ", pcd5)

a = np.reshape(meanstrain[0],(16,20))
b = np.reshape(meanstrain[1],(16,20))
c = np.reshape(meanstrain[2],(16,20))
d = np.reshape(meanstrain[3],(16,20))
e = np.reshape(meanstrain[4],(16,20))



f, position = plt.subplots(1,5)
position[0].imshow(a.T, cmap=plt.cm.binary)
position[1].imshow(b.T, cmap=plt.cm.binary)
position[2].imshow(c.T, cmap=plt.cm.binary)
position[3].imshow(d.T, cmap=plt.cm.binary)
position[4].imshow(e.T, cmap=plt.cm.binary)
plt.show()

def score(images,mean,c):
    return np.sum( images*safelog(mean) +  (1 - images)*safelog(1-mean)) + safelog(c)
    
aaa=0
ypredtrain = []
for j in range(len(imagestrain[:])):
    idx = 0
    sc_max = -9999999999999999999
    for i in range(5):
        sc = score(imagestrain[aaa], meanstrain[i],classpriors[i])
        if sc > sc_max:
            idx = i
            sc_max = sc
    ypredtrain.append(idx)
    aaa = aaa + 1

ypredtrain = np.array(ypredtrain) + 1

bbb=0
ypredtest = []
for j in range(len(imagestest[:])):
    idx2 = 0
    sc_max2 = -9999999999999999999
    for i in range(5):
        sc2 = score(imagestest[bbb], meanstrain[i],classpriors[i])
        if sc2 > sc_max2:
            idx2 = i
            sc_max2 = sc2
    ypredtest.append(idx2)
    bbb = bbb + 1

ypredtest = np.array(ypredtest) + 1



confusion_matrix = pd.crosstab(ypredtrain, ytruth, rownames = ['y_pred'], colnames = ['ytrain'])
print(confusion_matrix)

confusion_matrix = pd.crosstab(ypredtest, ytruthtest, rownames = ['y_pred'], colnames = ['ytrain'])
print(confusion_matrix)

