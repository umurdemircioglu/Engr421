import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


labelstrain = []
labelstest = []
imagestrain = np.zeros((1,320))
imagestest = np.zeros((1,320))

labels = pd.read_csv("hw02_data_set_labels.csv",header = None).to_numpy()
images = pd.read_csv("hw02_data_set_images.csv",header = None).to_numpy()

ytruth = np.concatenate((np.repeat(1, 25), np.repeat(2,25), np.repeat(3, 25),np.repeat(4, 25),np.repeat(5, 25)))
ytruthtest = np.concatenate((np.repeat(1, 14), np.repeat(2,14), np.repeat(3, 14),np.repeat(4, 14),np.repeat(5, 14)))

Y_truth = np.zeros((125, 5)).astype(int)
Y_truth[range(125), ytruth - 1] = 1

Y_truthtest = np.zeros((70, 5)).astype(int)
Y_truthtest[range(70), ytruthtest - 1] = 1

for i in range(5):
    labelstrain = np.append(labelstrain,labels[i*39:i*39+25, :])
    labelstest = np.append(labelstest, labels[i*39+25:i*39+39, :])
    imagestrain = np.vstack([imagestrain, images[i*39:i*39+25, :]])
    imagestest = np.vstack([imagestest, images[i*39+25:i*39+39, :]])

imagestrain = np.delete(imagestrain, 0, 0)
imagestest= np.delete(imagestest, 0, 0)

def sigmoid(images, w, w0):
    return(1 / (1 + np.exp(-(np.matmul(images, w) + w0))))

def gradient_W(images, y_truth, y_predicted):
    return np.asarray([-np.sum(np.repeat((Y_truth[:,c] - Y_predicted[:,c])[:, None], images.shape[1], axis = 1) * images, axis = 0) for c in range(5)]).transpose()

def gradient_w0(Y_truth, Y_predicted):
    return -np.sum(Y_truth - Y_predicted, axis = 0)

eta = 0.01
epsilon = 1e-3

W = np.random.uniform(low = -0.01, high = 0.01, size = (imagestrain.shape[1], 5))
w0 = np.random.uniform(low = -0.01, high = 0.01, size = (1, 5))
iteration = 1
objective_values = []
while 1:
    Y_predicted = sigmoid(imagestrain, W, w0)
    objective_values.append(np.sum((Y_truth - Y_predicted)**2)*(0.5))
    W_old = W
    w0_old = w0
    W = W - eta * gradient_W(imagestrain, Y_truth, Y_predicted)
    w0 = w0 - eta * gradient_w0(Y_truth, Y_predicted)
    if np.sqrt(np.sum((w0 - w0_old))**2 + np.sum((W - W_old)**2)) < epsilon:
        break
    iteration = iteration + 1
print(W)
print(w0)

y_predicted = np.argmax(Y_predicted, axis = 1) + 1
confusion_matrix = pd.crosstab(y_predicted, ytruth, rownames = ['y_pred'], colnames = ['ytrain'])
print(confusion_matrix)

ypredstest = sigmoid(imagestest,W,w0)
y = np.argmax(ypredstest, axis = 1) + 1
confusion_matrix = pd.crosstab(y, ytruthtest, rownames = ['y_pred'], colnames = ['ytest'])
print(confusion_matrix)

plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()
