import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('inputTotal.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Now we split the image to 140 firmas, each 250x250 size
cells = [np.hsplit(row,4) for row in np.vsplit(gray,35)]

# Make it into a Numpy array. It size will be (4,35,250,250)
x = np.array(cells)

# Now we prepare train_data and test_data. | 62500 = 250x250 imagen
train = x[:,:3].reshape(-1,62500).astype(np.float32) # Size = (15,62500) 15 firmas entrenar
test = x[:,3:4].reshape(-1,62500).astype(np.float32) # Size = (5,62500)  5 firmas prueba

#print "TRAIN: " + str(len(train))
#print "TEST: " + str(len(test))

# Create labels for train and test data
k = np.arange(7) #esta es la cantidad de personas/clases
train_labels = np.repeat(k,15)[:,np.newaxis] #15 firmas entrenar
test_labels = np.repeat(k,5)[:,np.newaxis]   #5 firmas prueba

#print "TRAIN_LABELS: " + str(len(train_labels))
#print "TEST_LABELS: " + str(len(test_labels))

# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.KNearest()
knn.train(train,train_labels)
ret,result,neighbours,dist = knn.find_nearest(test,k=5) #entre mas bajo el k, mas margen de error

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print str(accuracy)+"%"



# save the data
np.savez('knn_data.npz',train=train, train_labels=train_labels)

# Now load the data
with np.load('knn_data.npz') as data:
    print data.files
    train = data['train']
    train_labels = data['train_labels']
