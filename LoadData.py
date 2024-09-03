import numpy as np
import pickle
import os

x_load = []
y_load = []

count = 0
files = os.listdir("data")
for file in files:
    x=np.load(f"data/{file}")
    x = x.astype('float32') / 255
    x = x[0:10000, :] # no idea what last : colon does
    y = [count for _ in range(10000)]
    y = np.array(y).astype('float32')
    y = y.reshape(y.shape[0], 1) # basically changes y to array of arrays
    x_load.append(x)
    y_load.append(y)
    count += 1


features = np.array(x_load).astype('float32')
labels = np.array(y_load).astype('float32')
features=features.reshape(features.shape[0]*features.shape[1],28, 28, 1) # shape 0 - no of total classes, shape 1 - no of images, shape 2 - rows, shape 3- cols, shape 4 - pixels
labels=labels.reshape(labels.shape[0]*labels.shape[1],labels.shape[2])

with open("features", "wb") as f:
    pickle.dump(features, f)
with open("labels", "wb") as f:
    pickle.dump(labels, f)
