import cv2
import os
import numpy as np
import sys
import argparse

_cte_relacionMaxAnchoAlto = 1.5
_cte_relacionMinAnchoAlto = 0.5
_colorVerdeCaja = (0, 255, 0)
mascaraMediaPrecaucion = cv2.imread("./mascaras/mascaraMediaPrecaucion.ppm")
mascaraMediaProhibicion = cv2.imread("./mascaras/mascaraMediaProhibicion.ppm")
mascaraMediaSTOP = cv2.imread("./mascaras/mascaraMediaSTOP.ppm")

mascaraMediaProhibicion = cv2.cvtColor(mascaraMediaProhibicion, cv2.COLOR_BGR2GRAY)
mascaraMediaPrecaucion = cv2.cvtColor(mascaraMediaPrecaucion, cv2.COLOR_BGR2GRAY)
mascaraMediaSTOP = cv2.cvtColor(mascaraMediaSTOP, cv2.COLOR_BGR2GRAY)
# constructor mser
_delta = 10
_min_area = 100
_max_area = 2000
_max_variation = 0.2
_min_diversity = 0.2
_max_evolution = 200
_area_threshold = 1.01
_min_margin = 0.003
_edge_blur_size = 5
rects = []
# Rojos:
rojo_bajos1 = np.array([0, 50, 50], dtype=np.uint8)
rojo_bajos2 = np.array([240, 50, 50], dtype=np.uint8)
rojo_altos1 = np.array([12, 255, 255], dtype=np.uint8)
rojo_altos2 = np.array([256, 255, 255], dtype=np.uint8)

listaDir = os.listdir("/home/misial/Descargas/train/")
listaDir.sort()

def crea_compara_mascaras():
    for i in range(0, len(rects)):
        #print(str(i + 1) + "pinta mascaras")
        (x, y, w, h) = rects[i]
        if (x > 0) and (y > 0) and (y + h > 0):
            crop_img = img[y:y + h, x:x + w]
            # No sabemos cómo se hace para ajustar la imagen a 25 píxeles
            # print(str(x)+","+str(y)+","+str(w)+","+str(h))
            img_recortada = cv2.resize(crop_img, (25, 25), interpolation=cv2.INTER_NEAREST)
            hsv = cv2.cvtColor(img_recortada, cv2.COLOR_BGR2HSV)

            # Crear las mascaras
            mascara_rojo1 = cv2.inRange(hsv, rojo_bajos1, rojo_altos1)
            mascara_rojo2 = cv2.inRange(hsv, rojo_bajos2, rojo_altos2)

            # Juntar todas las mascaras
            mask = cv2.add(mascara_rojo1, mascara_rojo2)

            # comparacion de mascaras
            aux_mask_pro = (mask * mascaraMediaProhibicion)
            aux_mask_pre = (mask * mascaraMediaPrecaucion)
            aux_mask_stp = (mask * mascaraMediaSTOP)

            pix_mask_pro = np.sum(aux_mask_pro)
            pix_mask_pre = np.sum(aux_mask_pre)
            pix_mask_stp = np.sum(aux_mask_stp)

            # print("Pro: " + str(pix_mask_pro))
            # print("Pre: " + str(pix_mask_pre))
            # print("Stop: " + str(pix_mask_stp))
            if (pix_mask_pre > 150) and (pix_mask_pro > 150) and (pix_mask_stp > 150):
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


def regiones_detectadas():

    mser = cv2.MSER_create(_delta, _min_area, _max_area, _max_variation, _min_diversity, _max_evolution,
                           _area_threshold, _min_margin, _edge_blur_size)
    regions, _ = mser.detectRegions(grayEqHist)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    # pintamos los rectangulos de las regiones detectadas
    for j in range(0, len(regions)):
        x, y, w, h = cv2.boundingRect(regions[j])
        relacion = w / h
        if (relacion < _cte_relacionMaxAnchoAlto) and (relacion > _cte_relacionMinAnchoAlto):
            # print(len(filtrado_rects))
            rect1 = (x - (w // 10), y - (w // 10), w + (w // 10), h + (w // 10))
            rects.append(rect1)
            rect2 = (x - (w // 4), y - (w // 4), w + (w // 4), h + (w // 4))
            rects.append(rect2)
            cv2.rectangle(vis, (x - (w // 10), y - (w // 10)), (x + w + (w // 10), y + h + (w // 10)), _colorVerdeCaja)
            cv2.rectangle(vis, (x - (w // 4), y - (w // 4)), (x + w + (w // 4), y + h + (w // 4)), _colorVerdeCaja)


def main():
    global img, vis, grayEqHist
    for archivo in listaDir:
        # print(str(num) + "archivo")
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
        regiones_detectadas()

        print(archivo.title())
        cv2.imshow('detecciones', vis)
        cv2.waitKey()
        # print("longitud rects=" + str(len(rects)))

        crea_compara_mascaras()


main()


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