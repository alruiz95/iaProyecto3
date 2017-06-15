import numpy as np
import cv2
from matplotlib import pyplot as plt

#img = cv2.imread('input2.png')   #descomentar para probar otro caso
img = cv2.imread('input3.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Now we split the image to 20 cells, each 250x250 size
cells = [np.hsplit(row,4) for row in np.vsplit(gray,5)]

# Make it into a Numpy array. It size will be (4,5,250,250)
x = np.array(cells)

# Now we prepare train_data and test_data.
train = x[:,:3].reshape(-1,62500).astype(np.float32) # Size = (15,62500)
test = x[:,3:4].reshape(-1,62500).astype(np.float32) # Size = (5,62500)

print "TRAIN: " + str(len(train))
print "TEST: " + str(len(test))

# Create labels for train and test data
k = np.arange(1)
train_labels = np.repeat(k,15)[:,np.newaxis]
test_labels = np.repeat(k,5)[:,np.newaxis]
#test_labels = train_labels.copy()

print "TRAIN_LABELS: " + str(len(train_labels))
print "TEST_LABELS: " + str(len(test_labels))

# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.KNearest()
knn.train(train,train_labels)
ret,result,neighbours,dist = knn.find_nearest(test,k=1)

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print accuracy



# save the data
np.savez('knn_data.npz',train=train, train_labels=train_labels)

# Now load the data
with np.load('knn_data.npz') as data:
    print data.files
    train = data['train']
    train_labels = data['train_labels']
