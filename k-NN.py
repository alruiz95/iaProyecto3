# -*- coding: cp1252 -*-
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt

flag = False #no se ha realizado ningun entrenamiento

while True:
    print("------------------------------------------------")
    print("")
    img = cv2.imread('data/fondo negro/inputTotal.png')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Now we split the image to 140 firmas, each 250x250 size
    cells = [np.hsplit(row,4) for row in np.vsplit(gray,35)]

    # Make it into a Numpy array. It size will be (4,35,250,250)
    x = np.array(cells)

    train = []
    train_labels = []
    
    # Now we prepare train_data and test_data. | 62500 = 250x250 imagen
    if flag == True:
        # Now load the data
        with np.load('knn_data.npz') as data:
            #print data.files
            print ">> Información de entrenamiento cargada"
            print("")
            train = data['train']
            train_labels = data['train_labels']
    if flag == False:       
        train = x[:,:3].reshape(-1,62500).astype(np.float32) # Size = (15,62500) 15 firmas entrenar

    test = x[:,3:4].reshape(-1,62500).astype(np.float32) # Size = (5,62500)  5 firmas prueba

    # Create labels for train and test data
    k = np.arange(7) #esta es la cantidad de personas/clases

    if flag == False:
        train_labels = np.repeat(k,15)[:,np.newaxis] #15 firmas entrenar
        flag = True

    test_labels = np.repeat(k,5)[:,np.newaxis]   #5 firmas prueba

    # Initiate kNN, train the data, then test it with test data for k=1
    print("COMIENZA ENTRENAMIENTO")
    knn = cv2.KNearest()
    start_time = time.time()
    knn.train(train,train_labels)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("")
    print("COMIENZA PRUEBAS")
    start_time = time.time()
    ret,result,neighbours,dist = knn.find_nearest(test,k=5) #entre mas bajo el k, mas margen de error
    print("--- %s seconds ---" % (time.time() - start_time))
    print("")
        
    # Now we check the accuracy of classification
    # For that, compare the result with test_labels and check which are wrong
    print ("CALCULANDO RESULTADO")
    matches = result==test_labels
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/result.size
    print "--- " + str(accuracy)+"% de las firmas reconocidas ---"
    print("")

    # save the data
    print ("GUARDANDO INFORMACIÓN")
    print("--- Información guardada con éxito! ---")
    np.savez('knn_data.npz',train=train, train_labels=train_labels)
    print("")

    exiT = raw_input('Presione <q> para salir / cualquier tecla continuar: ')
    if exiT == 'q':
        break
    print("")
