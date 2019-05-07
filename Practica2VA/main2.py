#!/usr/bin/env python
import argparse
import os
import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Parametros HOGDescriptor
_winSize = (30, 30)
_blockSize = (15, 15)
_blockStride = (5, 5)
_cellSize = (5, 5)
_nbins = 9

_numeroImagenes = 846
_tamanyoHOG = 1296

def main():
    # Captacion de argumentos consola

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", "--train_path", help="Ruta carpeta imagenes de entrenamiento")
    parser.add_argument("--test_path", "--test_path", help="Ruta carpeta imagenes test")
    parser.add_argument("--detector", "--detector", help="Detector deseado")

    args = parser.parse_args()

    entrenamiento = args.train_path
    test = args.test_path
    train = args.train_path
    # Cargado de todas las imagenes de la ruta a testear
    lista_dir = os.listdir(train)
    lista_dir.sort()

    matrizCarac = np.zeros((1, 1296))
    matrizEtiq = np.zeros((_numeroImagenes, 1))
    contadorImagenes = 0

    for carpeta in lista_dir:
        ruta_actual = os.path.join(train, carpeta)
        lista_img = os.listdir(ruta_actual)
        lista_img.sort()

        #print("estoy en carpeta: " + carpeta.title())
        for archivo in lista_img:

            titulo = archivo.title().lower()
            #print(titulo)
            # Cargamos imagen
            img = cv2.imread(os.path.join(ruta_actual, titulo))

            # Redimensionamos la imagen a 30x30
            img_recortada = cv2.resize(img, (30, 30), interpolation=cv2.INTER_LINEAR)

            # Convertimos a escala de grises
            gray = cv2.cvtColor(img_recortada, cv2.COLOR_BGR2GRAY)

            # Equalizamos hist. para mejorar contraste (detecta mas senyales usando esta imagen que la escala grises normal)
            gray_eq = cv2.equalizeHist(gray)

            # Calculamos HOG
            hog = cv2.HOGDescriptor(_winSize, _blockSize, _blockStride, _cellSize, _nbins)
            calculoHOG = hog.compute(gray_eq)

            #Insertamos DescripcionHOG en la matriz
            #print(np.transpose(calculoHOG).shape)

            # si es la primera fila, sustituye la inicializada a 0
            if contadorImagenes == 0:
                matrizCarac[0] = np.transpose(calculoHOG)
            else:
                matrizCarac = np.concatenate((matrizCarac, np.transpose(calculoHOG)))
            matrizEtiq[contadorImagenes, 0] = str(carpeta.title())
            contadorImagenes = contadorImagenes + 1
            #print(calculoHOG)
            #print(len(calculoHOG))

    print("Numero imagenes train totales =" + str(contadorImagenes))
    print(matrizCarac)
    print(matrizCarac.shape)
    print(".----------------------------")
    print(matrizEtiq)
    print(matrizEtiq.shape)


main()
print("finalizado")
# tam calucloHOG= 1296 --- mitad = 648
'''
# Calculamos LDA

X = calculoHOG

y_a = np.zeros(648,3)
y_b = np.ones(648,3)
y = np.concatenate((y_a, y_b))

lda = LinearDiscriminantAnalysis()
lda.fit(X, y)

calculoLDA = lda.transform(X)

print("-----------------------")
print(calculoLDA)
'''