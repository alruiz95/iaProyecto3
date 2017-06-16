from PIL import Image
import glob
import random
from resizeimage import resizeimage
from sklearn.neural_network import MLPClassifier
import pickle
import time
import os


todas_firmas = []

def gray(img):
    width, height = img.size
    setbites = []
    pix = img.load()
    for y in range(0, height):
        for x in range(0, width):
            if pix[x,y]== 0:
                setbites.append(1)
            else:
                setbites.append(0)
    return setbites



def main():
    folders =  os.walk("data/fondo blanco")
    firmas = next(folders)[1]
    cont = 0;
    for carpetaFirma in firmas:
        image_list = []
        salidaNew=[]
        for filename in glob.glob('data/fondo blanco/'+carpetaFirma+'/*.jpg'):  # assuming gif
            im = Image.open(filename)
            im = im.convert('1')
            im = resizeimage.resize_cover(im, [20, 20])
            matriz = gray(im)
            image_list.append(matriz)
        for x in range(0, len(firmas)):
            if x==cont:
                salidaNew.append(1)
            else:
                salidaNew.append(0)
        salidaNew.append(0)
        elemento = [salidaNew,image_list]
        todas_firmas.append(elemento)

    random.shuffle(todas_firmas)
    entradas = []
    salidas = []
    for mae in todas_firmas:
        for firma in mae[1]:
            entradas.append(firma)
            salidas.append(mae[0])


    capaEntrada = len(entradas[0])
    capaOculta = capaEntrada+3
    capaSalida = len(firmas)+1


    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(capaOculta,), random_state=1)

    print("COMIENZA ENTRENAMIENTO")
    print("")
    start_time = time.time()
    clf.fit(entradas, salidas)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("")
    filename = 'finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    print("Se ha guardado el entrenamiento correctamente ")
    print("RESULTADO")
    print ("Resultado calculado:")
    start_time = time.time()
    print clf.predict([entradas[0]])
    print("--- %s seconds ---" % (time.time() - start_time))
    print ("Resultado deseado:")
    print salidas[0]
    print("")


main()
"""
img = Image.open('image.jpg')
img = img.convert('1')
img.save('gray.jpg')
print img.size
print gray(img)"""