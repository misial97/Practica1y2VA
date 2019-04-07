import cv2
import os
import numpy as np
import sys
import argparse

_cte_relacionMaxAnchoAlto = 1.5
_cte_relacionMinAnchoAlto = 0.5
_colorVerdeCaja = (0, 255, 0)
mascaraMediaPrecaucion = cv2.imread("/home/misial/PycharmProjects/practica2/mascaras/mascaraMediaPrecaucion.ppm")
mascaraMediaProhibicion = cv2.imread("/home/misial/PycharmProjects/practica2/mascaras/mascaraMediaProhibicion.ppm")
mascaraMediaSTOP = cv2.imread("/home/misial/PycharmProjects/practica2/mascaras/mascaraMediaSTOP.ppm")

mascaraMediaProhibicion = cv2.cvtColor(mascaraMediaProhibicion, cv2.COLOR_BGR2GRAY)
mascaraMediaPrecaucion = cv2.cvtColor(mascaraMediaPrecaucion, cv2.COLOR_BGR2GRAY)
mascaraMediaSTOP = cv2.cvtColor(mascaraMediaSTOP, cv2.COLOR_BGR2GRAY)
# constructor mser
_delta = 7
_min_area = 100
_max_area = 2000
_max_variation = 0.2
_min_diversity = 0.2
_max_evolution = 200
_area_threshold = 1.01
_min_margin = 0.003
_edge_blur_size = 5

cv2.imshow("mascaramediaconstante", mascaraMediaPrecaucion)
cv2.waitKey()
listaDir = os.listdir("/home/misial/Descargas/train/")
listaDir.sort()
print(listaDir[0].title())
for archivo in listaDir:

    # cargamos imagen
    img = cv2.imread("/home/misial/Descargas/train/" + archivo.title().lower())
    # cv2.imshow("original", img)
    vis = img.copy()
    # cv2.waitKey()

    # convertimos a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("escala_grises",gray)
    # cv2.waitKey()

    # equalizamos histograma para mejorar contraste (detecta más señales usando esta imagen que la escala grises normal)
    grayEqHist = cv2.equalizeHist(gray)
    # cv2.imshow("hist", grayEqHist)
    # cv2.waitKey()

    # detectamos regiones alto contraste
    mser = cv2.MSER_create(_delta, _min_area, _max_area, _max_variation, _min_diversity, _max_evolution,
                           _area_threshold, _min_margin, _edge_blur_size)
    regions, _ = mser.detectRegions(grayEqHist)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    filtrado_rects = []
    rects = []

    # pintamos los rectangulos de las regiones detectadas
    for i in range(0, len(regions)):
        x, y, w, h = cv2.boundingRect(regions[i])
        relacion = w / h
        if (relacion < _cte_relacionMaxAnchoAlto) and (relacion > _cte_relacionMinAnchoAlto):
            # print(len(filtrado_rects))
            rect1 = (x - (w // 10), y - (w // 10), w + (w // 10), h + (w // 10))
            rects.append(rect1)
            rect2 = (x - (w // 4), y - (w // 4), w + (w // 4), h + (w // 4))
            rects.append(rect2)
            cv2.rectangle(vis, (x - (w // 10), y - (w // 10)), (x + w + (w // 10), y + h + (w // 10)), _colorVerdeCaja)
            cv2.rectangle(vis, (x - (w // 4), y - (w // 4)), (x + w + (w // 4), y + h + (w // 4)), _colorVerdeCaja)
    print(archivo.title())
    cv2.imshow('detecciones', vis)
    cv2.waitKey()

    # habria que continuar por "2 Utilizar el espacio de color HSV para localizar los píxeles que sean de color rojo"
    # Rojos:
    rojo_bajos1 = np.array([0, 50, 50], dtype=np.uint8)
    rojo_bajos2 = np.array([240, 50, 50], dtype=np.uint8)
    rojo_altos1 = np.array([12, 255, 255], dtype=np.uint8)
    rojo_altos2 = np.array([256, 255, 255], dtype=np.uint8)

    for i in range(0, len(rects)):
        (x, y, w, h) = rects[i]
        crop_img = img[y:y + h, x:x + w]
        # No sabemos cómo se hace para ajustar la imagen a 25 píxeles
        img_recortada = cv2.resize(crop_img, (25, 25), interpolation=cv2.INTER_NEAREST)
        hsv = cv2.cvtColor(img_recortada, cv2.COLOR_BGR2HSV)

        # Crear las mascaras
        mascara_rojo1 = cv2.inRange(hsv, rojo_bajos1, rojo_altos1)
        mascara_rojo2 = cv2.inRange(hsv, rojo_bajos2, rojo_altos2)

        # Juntar todas las mascaras
        mask = cv2.add(mascara_rojo1, mascara_rojo2)

        #comparacion de mascaras

        aux_mask_pro = (mask * mascaraMediaProhibicion)
        print("Pro: " + str(np.sum(aux_mask_pro)))
        aux_mask_pre = (mask * mascaraMediaPrecaucion)
        print("Pre: " + str(np.sum(aux_mask_pre)))
        aux_mask_stp = (mask * mascaraMediaSTOP)
        print("Stop: " + str(np.sum(aux_mask_stp)))

    # comparar mascara con mascara media (no sabemos si asi puesta la media funciona)
    # nombre = "Mascara" + str(i)
    # cv2.imshow(nombre, mask)
    # cv2.waitKey()
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