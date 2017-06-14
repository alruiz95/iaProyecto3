from PIL import Image

def gray(img):
    width, height = img.size
    matrix = []
    pix = img.load()
    for y in range(0, height):
        elemento = []
        for x in range(0, width):
            if pix[x,y]== 0:
                elemento.append(1)
            else:
                elemento.append(0)
        matrix.append(elemento)
    return matrix

img = Image.open('image.jpg')
img = img.convert('1')
img.save('gray.jpg')
print img.size
print gray(img)