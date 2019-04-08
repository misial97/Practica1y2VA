#!/usr/bin/env python
import cv2
import os
import numpy as np
import sys

_cte_relacionMaxAnchoAlto = 1.5
_cte_relacionMinAnchoAlto = 0.5
_colorVerdeCaja = (0, 255, 0)

# Parametros constructor MSER
_delta = 10
_min_area = 100
_max_area = 2000
_max_variation = 0.2
_min_diversity = 0.2
_max_evolution = 200
_area_threshold = 1.01
_min_margin = 0.003
_edge_blur_size = 5

# Rojos:
_rojo_bajos1 = np.array([0, 50, 50], dtype=np.uint8)
_rojo_bajos2 = np.array([240, 50, 50], dtype=np.uint8)
_rojo_altos1 = np.array([12, 255, 255], dtype=np.uint8)
_rojo_altos2 = np.array([256, 255, 255], dtype=np.uint8)


def main():
    entrenamiento = sys.argv[1]
    test = sys.argv[2]
   # detector = sys.argv[3]

    fichero_result = test + "/resultado.txt"

    mascaras_medias = mascara_media(entrenamiento)

    lista_dir = os.listdir(test)
    lista_dir.sort()

    for archivo in lista_dir:
        titulo = archivo.title().lower()

        # Cargamos imagen
        img = cv2.imread(test + "/" + titulo)

        vis = img.copy()

        # Convertimos a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Equalizamos hist. para mejorar contraste (detecta mas senyales usando esta imagen que la escala grises normal)
        gray_eq = cv2.equalizeHist(gray)

        # Detectamos regiones alto contraste
        rects = regiones_detectadas(gray_eq, vis)

        datos = crea_compara_mascaras(img, mascaras_medias, rects)

        for senyal in datos:
            escribir(fichero_result, titulo, senyal)


def crea_compara_mascaras(img, mascaras_medias, rects):
    datos = [[]]
    senyal = []
    for i in range(0, len(rects)):
        (x, y, w, h) = rects[i]
        if (x > 0) and (y > 0) and (y + h > 0):
            crop_img = img[y:y + h, x:x + w]

            # No sabemos como se hace para ajustar la imagen a 25 pixeles
            img_recortada = cv2.resize(crop_img, (25, 25), interpolation=cv2.INTER_NEAREST)
            hsv = cv2.cvtColor(img_recortada, cv2.COLOR_BGR2HSV)

            # Crear las mascaras
            mascara_rojo1 = cv2.inRange(hsv, _rojo_bajos1, _rojo_altos1)
            mascara_rojo2 = cv2.inRange(hsv, _rojo_bajos2, _rojo_altos2)

            # Juntar todas las mascaras
            mask = cv2.add(mascara_rojo1, mascara_rojo2)

            # Comparacion de mascaras
            aux_mask_pro = (mask * mascaras_medias[0])
            aux_mask_pre = (mask * mascaras_medias[1])
            aux_mask_stp = (mask * mascaras_medias[2])

            pix_mask_pro = np.sum(aux_mask_pro)
            pix_mask_pre = np.sum(aux_mask_pre)
            pix_mask_stp = np.sum(aux_mask_stp)

           # print(str(i) + ";" + str(pix_mask_pro) + ";" + str(pix_mask_pre) + ";" + str(pix_mask_stp))

            if (pix_mask_pre > 100) and (pix_mask_pro > 100) and (pix_mask_stp > 100):
                if (pix_mask_pro > pix_mask_pre) and (pix_mask_pro > pix_mask_stp):
                    print("prohibicion detectada 1")
                    senyal.append(x)
                    senyal.append(y)
                    senyal.append(x + w)
                    senyal.append(y + h)
                    senyal.append(1)
                    datos.append(senyal)
                    senyal = []
                elif (pix_mask_pre > pix_mask_pro) and (pix_mask_pre > pix_mask_stp):
                    print("peligro detectada 2")
                    senyal.append(x)
                    senyal.append(y)
                    senyal.append(x + w)
                    senyal.append(y + h)
                    senyal.append(2)
                    datos.append(senyal)
                    senyal = []
                elif (pix_mask_stp > pix_mask_pre) and (pix_mask_stp > pix_mask_pro):
                    print("stop detectado 3")
                    senyal.append(x)
                    senyal.append(y)
                    senyal.append(x + w)
                    senyal.append(y + h)
                    senyal.append(3)
                    datos.append(senyal)
                    senyal = []

            '''
                        if (pix_mask_pre > 150) and (pix_mask_pro > 10) and (pix_mask_stp > 10):
                nombre = "Mascara" + str(i)
                cv2.imshow(nombre, mask)
                cv2.waitKey()
            if (pix_mask_pre > 100) and (pix_mask_pro > 100) and (pix_mask_stp > 100):
                if (pix_mask_pro > pix_mask_pre) and (pix_mask_pro > pix_mask_stp):
                    cv2.imshow("PROHIBICION", mask)
                    cv2.waitKey()
                elif (pix_mask_pre > pix_mask_pro) and (pix_mask_pre > pix_mask_stp):
                    cv2.imshow("PRECAUCION", mask)
                    cv2.waitKey()
                elif (pix_mask_stp > pix_mask_pre) and (pix_mask_stp > pix_mask_pro):
                    cv2.imshow("STOP", mask)
                    cv2.waitKey()
            '''

    return datos


def regiones_detectadas(gray, vis):
    rects = []
    mser = cv2.MSER_create(_delta, _min_area, _max_area, _max_variation, _min_diversity, _max_evolution,
                           _area_threshold, _min_margin, _edge_blur_size)
    regions, _ = mser.detectRegions(gray)

    # Pintamos los rectangulos de las regiones detectadas
    for j in range(0, len(regions)):
        x, y, w, h = cv2.boundingRect(regions[j])
        relacion = w / h
        if (relacion < _cte_relacionMaxAnchoAlto) and (relacion > _cte_relacionMinAnchoAlto):
            rect1 = (x - (w // 10), y - (w // 10), w + (w // 10), h + (w // 10))
            rects.append(rect1)
            rect2 = (x - (w // 4), y - (w // 4), w + (w // 4), h + (w // 4))
            rects.append(rect2)
            cv2.rectangle(vis, (x - (w // 10), y - (w // 10)), (x + w + (w // 10), y + h + (w // 10)), _colorVerdeCaja)
            cv2.rectangle(vis, (x - (w // 4), y - (w // 4)), (x + w + (w // 4), y + h + (w // 4)), _colorVerdeCaja)

    return rects


def mascara_media(ruta):
    lista = os.listdir(ruta)
    lista.sort()

    mascaras = [[], [], []]
    parametros = []

    with open(ruta + "/gt.txt", "r") as fichero:
        for linea in fichero:
            parametros.append(linea.split(";"))
        parametros.pop() #eliminamos el ultimo elemento de la lista ("\r\n")
        #print(parametros)

        for datoSenyal in parametros:
            titulo = datoSenyal[0]
            x = int(datoSenyal[1])
            y = int(datoSenyal[2])
            w = int(datoSenyal[3]) - x
            h = int(datoSenyal[4]) - y
            tipo = int(datoSenyal[5].replace("\r\n", ""))
            #print(titulo+";"+str(x)+";"+str(y)+";"+str(w)+";"+str(h)+";"+str(tipo))

            # Prohibicion
            if ((tipo >= 0) and (tipo <= 5)) or ((tipo >= 7) and (tipo <= 10)) or ((tipo == 15) or (tipo == 16)):
                imagen = cv2.imread(ruta + "/" + titulo)
                # Se recorta la senyal y se redimensiona a 25x25
                recortada = cv2.resize(imagen[y:y + h, x:x + w], (25, 25), interpolation=cv2.INTER_NEAREST)
                hsv = cv2.cvtColor(recortada, cv2.COLOR_BGR2HSV)

                # Crear las mascaras
                mascara_bajos = cv2.inRange(hsv, _rojo_bajos1, _rojo_altos1)
                mascara_altos = cv2.inRange(hsv, _rojo_bajos2, _rojo_altos2)

                # Juntar todas las mascaras
                mascara_final = cv2.add(mascara_bajos, mascara_altos)

                mascaras[0].append(mascara_final)

            # Peligro
            elif (tipo == 11) or ((tipo >= 18) and (tipo <= 31)):
                imagen = cv2.imread(ruta + "/" + titulo)
                # Se recorta la senyal y se redimensiona a 25x25
                recortada = cv2.resize(imagen[y:y + h, x:x + w], (25, 25), interpolation=cv2.INTER_NEAREST)
                hsv = cv2.cvtColor(recortada, cv2.COLOR_BGR2HSV)

                # Crear las mascaras
                mascara_bajos = cv2.inRange(hsv, _rojo_bajos1, _rojo_altos1)
                mascara_altos = cv2.inRange(hsv, _rojo_bajos2, _rojo_altos2)

                # Juntar todas las mascaras
                mascara_final = cv2.add(mascara_bajos, mascara_altos)

                mascaras[1].append(mascara_final)

            # STOP
            elif tipo == 14:
                imagen = cv2.imread(ruta + "/" + titulo)
                # Se recorta la senyal y se redimensiona a 25x25
                recortada = cv2.resize(imagen[y:y + h, x:x + w], (25, 25), interpolation=cv2.INTER_NEAREST)
                hsv = cv2.cvtColor(recortada, cv2.COLOR_BGR2HSV)

                # Crear las mascaras
                mascara_bajos = cv2.inRange(hsv, _rojo_bajos1, _rojo_altos1)
                mascara_altos = cv2.inRange(hsv, _rojo_bajos2, _rojo_altos2)

                # Juntar todas las mascaras
                mascara_final = cv2.add(mascara_bajos, mascara_altos)

                mascaras[2].append(mascara_final)

    mascaras_media = [np.zeros((25, 25)), np.zeros((25, 25)), np.zeros((25, 25))]

    # Calculamos la media para STOP
    for mascara in mascaras[2]:
        mascaras_media[2] += mascara
    mascaras_media[2] = mascaras_media[2] // len(mascaras[2])
    condicion1 = mascaras_media[2] < 100
    condicion2 = mascaras_media[2] >= 100
    mascaras_media[2][condicion1] = 0
    mascaras_media[2][condicion2] = 255

    # Calculamos la media para Peligro (1) y Prohibicion(0)
    for mascara in mascaras[0]:
        mascaras_media[0] += mascara

    mascaras_media[0] = mascaras_media[0] // len(mascaras[0])
    condicion1 = mascaras_media[0] < 100
    condicion2 = mascaras_media[0] >= 100
    mascaras_media[0][condicion1] = 0
    mascaras_media[0][condicion2] = 255

    for mascara in mascaras[1]:
        mascaras_media[1] += mascara
    mascaras_media[1] = mascaras_media[1] // len(mascaras[1])
    condicion1 = mascaras_media[1] < 100
    condicion2 = mascaras_media[1] >= 100
    mascaras_media[1][condicion1] = 0
    mascaras_media[1][condicion2] = 255

    #mascaras_media[0] = cv2.cvtColor(mascaras_media[0], cv2.COLOR_BGR2GRAY)
    #mascaras_media[1] = cv2.cvtColor(mascaras_media[1], cv2.COLOR_BGR2GRAY)
    #mascaras_media[2] = cv2.cvtColor(mascaras_media[2], cv2.COLOR_BGR2GRAY)

    cv2.imshow("pro", mascaras_media[0])
    cv2.waitKey()
    cv2.imshow("pre", mascaras_media[1])
    cv2.waitKey()
    cv2.imshow("stp", mascaras_media[2])
    cv2.waitKey()

    return mascaras_media


def escribir(ruta, titulo, datos):
    linea = titulo
    for dato in datos:
        linea = linea + ";" + str(dato)

    fichero = open(ruta, "a")
    fichero.write(linea + "\n")
    fichero.close()


main()
print("FINALIZADO. FICHERO resultado.txt")

# comparar mascara con mascara media (no sabemos si asi puesta la media funciona)
# nombre = "Mascara" + str(i)
#    cv2.imshow("mascara", mask)
#   cv2.waitKey()
'''
                # Eliminar duplicidades
            if len(filtrado_rects) == 0:
                rect = (x, y, w, h)
                filtrado_rects.append(rect)
            else:
                esta = False
                for j in range(0, len(filtrado_rects)):
                    (coorX1, coorY1, anchura, altura) = filtrado_rects[j]
                    coorX2 = coorX1 + anchura
                    coorY2 = coorY1 + altura
                    condCoordInfIzq = (x > coorX1) and (x < coorX2) and (y > coorY1) and (y < coorY2)
                    condCoordSupDch = (x > coorX1) and (x < coorX2) and (y > coorY1) and (y < coorY2)
                    if condCoordInfIzq and condCoordSupDch:
                        esta = True
                if not esta :
                    # if height is enough
                    # create rectangle for bounding
                    rect = (x, y, w, h)
                    filtrado_rects.append(rect)
                    rect = (x-(w//10), y-(w//10), w+(w//10), h+(w//10))
                    rects.append(rect)
                    cv2.rectangle(vis, (x-(w//10), y-(w//10)), (x + w+(w//10), y + h+(w//10)), _colorVerdeCaja)
'''