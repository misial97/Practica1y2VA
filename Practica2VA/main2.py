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

# Constantes necesarias para crear arrays y matrices
_numeroImagenesTrain = 846
_tamanyoHOG = 1296


def main():

    # Captacion de argumentos consola
    detector, test, train = capta_args()

    #Comprobamos el detector a usar
    if detector == "LDA-BAYES":
        lda_bayes(test, train)
    else:
        print("Error. El dectector no existe, unicamente existe LDA-BAYES")


def lda_bayes(test, train):
    print("\nHa seleccionado LDA-BAYES.\n")

    # Inicializamos variables que necesitamos
    matrizCarac = np.zeros((1, _tamanyoHOG))
    matrizCaracTest = np.zeros((1, _tamanyoHOG))
    etiquetasImg = np.zeros(_numeroImagenesTrain)
    contadorImagenes = 0
    contadorImagenesTest = 0

    # Cargado de todas las imagenes de la ruta a testear
    lista_dir = os.listdir(train)
    lista_dir.sort()

    # Entrenamiento del clasificador
    print("Generando la matriz de caracteristicas de las imagenes de train y el vector de etiquetas...\n")
    matrizCarac, etiquetasImg = entrenaClasificador(contadorImagenes, etiquetasImg, lista_dir, matrizCarac, train)

    # Tratamiento imagenes de test
    print("Generando la matriz de caracteristicas de las imagenes de test...\n")
    lista_img_test, matrizCaracTest = lectura_imgs_test(contadorImagenesTest, matrizCaracTest, test)

    # Ya con las matrices de los descrp y de las etiquetas creadas --> LDA(reduc. dimensionalidad)
    # Ademas clasifica las imagenes de test
    print("Realizando LDA y prediccion...\n")
    prediccion = lda_clasifica(etiquetasImg, matrizCarac, matrizCaracTest)

    # Escritura del fichero con la prediccion del clasificador
    escribe_fichero(lista_img_test, prediccion)


def lectura_imgs_test(contadorImagenesTest, matrizCaracTest, test):
    # Obtenemos ruta carpeta test y una lista ordenada de sus imagenes
    ruta_actual_test = os.path.join(test)
    lista_img_test = os.listdir(test)
    lista_img_test.sort()

    # Eliminamos posibles archivos ocultos de la lista a recorrer (.directory)
    for img in lista_img_test:
        if img.title()[0] == ".":
            lista_img_test.remove(img)
    # En cada imagen calculamos su vector de caracteristicas con el previo tratamiento a la imagen
    for archivo in lista_img_test:
        calculoHOG = tratamiento_imgs(archivo, ruta_actual_test)

        # Insertamos DescripcionHOG en la matriz
        # si es la primera fila, sustituye la inicializada a 0
        if contadorImagenesTest == 0:
            matrizCaracTest[0] = np.transpose(calculoHOG)
        else:
            matrizCaracTest = np.concatenate((matrizCaracTest, np.transpose(calculoHOG)))
        contadorImagenesTest = contadorImagenesTest + 1
    return lista_img_test, matrizCaracTest


def entrenaClasificador(contadorImagenes, etiquetasImg, lista_dir, matrizCarac, train):
    # La estructura de carpetas de train hace que se necesite un for dentro de otro para recorrer todas las imagenes
    for carpeta in lista_dir:
        ruta_actual = os.path.join(train, carpeta)
        lista_img = os.listdir(ruta_actual)
        lista_img.sort()

        # En cada imagen calculamos su vector de caracteristicas con el previo tratamiento a la imagen
        for archivo in lista_img:

            calculoHOG = tratamiento_imgs(archivo, ruta_actual)

            # Insertamos DescripcionHOG en la matriz
            # si es la primera fila, sustituye la inicializada a 0
            if contadorImagenes == 0:
                matrizCarac[0] = np.transpose(calculoHOG)
            else:
                matrizCarac = np.concatenate((matrizCarac, np.transpose(calculoHOG)))
            etiquetasImg[contadorImagenes] = str(carpeta.title())
            contadorImagenes = contadorImagenes + 1
    return matrizCarac, etiquetasImg


def capta_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", "--train_path", help="Ruta carpeta imagenes de entrenamiento")
    parser.add_argument("--test_path", "--test_path", help="Ruta carpeta imagenes test")
    parser.add_argument("--detector", "--detector", help="Detector deseado (LDA-BAYES)")

    # Argumentos ruta de carpeta test, ruta de carpeta train y detector deseado
    args = parser.parse_args()
    test = args.test_path
    train = args.train_path
    detector = args.detector
    return detector, test, train


def tratamiento_imgs(archivo, ruta_actual):

    # Titulo de la imagen actual, para poder cargarla despues
    titulo = archivo.title().lower()
    ruta_imagen = os.path.join(ruta_actual, titulo)

    # Cargamos imagen
    img = cv2.imread(ruta_imagen)

    # Redimensionamos la imagen a 30x30
    img_recortada = cv2.resize(img, (30, 30), interpolation=cv2.INTER_LINEAR)

    # Convertimos a escala de grises
    gray = cv2.cvtColor(img_recortada, cv2.COLOR_BGR2GRAY)

    # Equalizamos hist. para mejorar contraste (detecta mas senyales usando esta imagen que la escala grises normal)
    gray_eq = cv2.equalizeHist(gray)

    # Calculamos HOG
    hog = cv2.HOGDescriptor(_winSize, _blockSize, _blockStride, _cellSize, _nbins)
    calculoHOG = hog.compute(gray_eq)

    return calculoHOG


def escribe_fichero(lista_img_test, prediccion):
    # El fichero se creara en la ruta donde se encuentre el main.py
    ruta_fichero = os.path.join(os.getcwd(), "resultado.txt")

    # Si ya existe, elimina el fichero
    print("Comprobacion de fichero existente...\n")
    if os.path.isfile(ruta_fichero):
        print("El fichero ya existia.\n    Eliminando fichero....\n")
        os.remove(ruta_fichero)
        print("Fichero eliminado.\n")
    else:
        print("El fichero no existe.\n")

    # Recorremos el array de la prediccion junto a la lista de imagenes test.
    print("Generando el fichero nuevo.\n")
    for i in range(0, len(prediccion)):
        tipoPredict = int(prediccion[i])

        # Comprobamos si es menor que 10 para poner el 0 delante como dice el formato
        if tipoPredict < 10:
            tipoPredictStr = "0" + str(tipoPredict)
        else:
            tipoPredictStr = str(tipoPredict)

        # Generamos la linea y la imprimimos en el fichero
        linea = lista_img_test[i] + "; " + tipoPredictStr + "\n"
        fichero = open(ruta_fichero, "a")
        fichero.write(linea)
        fichero.close()


def lda_clasifica(etiquetasImg, matrizCarac, matrizCaracTest):
    # Reduccion de la dimensionalidad y prediccion
    lda = LinearDiscriminantAnalysis()
    lda.fit(matrizCarac, etiquetasImg)
    prediccion = lda.predict(matrizCaracTest)

    return prediccion


main()
print("Fichero generado correctamente en: " + os.getcwd() + "\n")
