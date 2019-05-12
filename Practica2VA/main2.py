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
    parser.add_argument("--detector", "--detector", help="Detector deseado (LDA-BAYES)")

    args = parser.parse_args()

    test = args.test_path
    train = args.train_path
    detector = args.detector

    if detector == "LDA-BAYES":
        # Cargado de todas las imagenes de la ruta a testear
        lista_dir = os.listdir(train)
        lista_dir.sort()

        matrizCarac = np.zeros((1, 1296))
        matrizCaracTest = np.zeros((1, 1296))
        etiquetasImg = np.zeros(_numeroImagenes)
        contadorImagenes = 0
        contadorImagenesTest = 0

        # Entrenamiento del clasificador
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

                # Insertamos DescripcionHOG en la matriz
                # si es la primera fila, sustituye la inicializada a 0
                if contadorImagenes == 0:
                    matrizCarac[0] = np.transpose(calculoHOG)
                else:
                    matrizCarac = np.concatenate((matrizCarac, np.transpose(calculoHOG)))
                etiquetasImg[contadorImagenes] = str(carpeta.title())
                contadorImagenes = contadorImagenes + 1

        # Imagenes test
        ruta_actual_test = os.path.join(test)
        print(ruta_actual_test)
        lista_img_test = os.listdir(test)
        lista_img_test.sort()
        # Eliminamos posibles archivos ocultos de la lista a recorrer (.directory)
        for img in lista_img_test:
            if img.title()[0] == ".":
                lista_img_test.remove(img)

        for archivo in lista_img_test:
            titulo = archivo.title().lower()
            print(titulo)
            # print(titulo)
            # Cargamos imagen
            img = cv2.imread(os.path.join(ruta_actual_test, titulo))

            # Redimensionamos la imagen a 30x30
            img_recortada = cv2.resize(img, (30, 30), interpolation=cv2.INTER_LINEAR)

            # Convertimos a escala de grises
            gray = cv2.cvtColor(img_recortada, cv2.COLOR_BGR2GRAY)

            # Equalizamos hist. para mejorar contraste (detecta mas senyales usando esta imagen que la escala grises normal)
            gray_eq = cv2.equalizeHist(gray)

            # Calculamos HOG
            hog = cv2.HOGDescriptor(_winSize, _blockSize, _blockStride, _cellSize, _nbins)
            calculoHOG = hog.compute(gray_eq)

            # Insertamos DescripcionHOG en la matriz
            # si es la primera fila, sustituye la inicializada a 0
            if contadorImagenesTest == 0:
                matrizCaracTest[0] = np.transpose(calculoHOG)
            else:
                matrizCaracTest = np.concatenate((matrizCaracTest, np.transpose(calculoHOG)))
            contadorImagenesTest = contadorImagenesTest + 1

        # Ya con las matrices de los descrp y de las etiquetas creadas --> LDA(reduc. dimensionalidad)
        lda = LinearDiscriminantAnalysis()
        lda.fit(matrizCarac, etiquetasImg)

        # calculoLDA = lda.transform(matrizCarac)
        prediccion = lda.predict(matrizCaracTest)

        # print(prediccion)

        ruta_fichero = os.path.join(os.getcwd(), "resultado.txt")
        # Si ya existe, elimina el fichero
        if os.path.isfile(ruta_fichero):
            print("Eliminando fichero....")
            os.remove(ruta_fichero)

        for i in range(0, len(prediccion)):
            tipoPredict = int(prediccion[i])
            if tipoPredict < 10:
                tipoPredictStr = "0" + str(tipoPredict)
            else:
                tipoPredictStr = str(tipoPredict)

            linea = lista_img_test[i] + "; " + tipoPredictStr + "\n"
            fichero = open(ruta_fichero, "a")
            fichero.write(linea)
            fichero.close()
    else:
        print("Error. El dectector no existe, unicamente existe LDA-BAYES")


main()
print("finalizado")
