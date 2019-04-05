import cv2
import os
import numpy as np

_numCarpetas = 31  # precaucion
#_numCarpetas = 12 # prohibicion
mascaras = []
mascara_media = np.zeros((25, 25))
print(type(mascara_media))
mascara_media.tolist()

#Rojos:
rojo_bajos1 = np.array([0,50,50], dtype=np.uint8)
rojo_bajos2 = np.array([240,50,50], dtype=np.uint8)
rojo_altos1 = np.array([12, 255, 255], dtype=np.uint8)
rojo_altos2 = np.array([256, 255, 255], dtype=np.uint8)

#for numCarp in range(0, _numCarpetas):
for numCarp in range(17, _numCarpetas):
    if numCarp < 10:
        carpeta = "0" + str(numCarp)
    else:
        carpeta = str(numCarp)

    ruta = "/home/misial/Descargas/train_recortadas/precaucion/" + carpeta
    #ruta = "/home/misial/Descargas/train_recortadas/prohib/" + carpeta
    contenido = os.listdir(ruta)
    contenido.sort()

    for archivo in contenido:
        # cargamos imagen
        img = cv2.imread(ruta + "/" + archivo.title().lower())
        redimension = cv2.resize(img, (25, 25), interpolation=cv2.INTER_NEAREST)
        hsv = cv2.cvtColor(redimension, cv2.COLOR_BGR2HSV)

        # Crear las mascaras
        mascara_rojo1 = cv2.inRange(hsv, rojo_bajos1, rojo_altos1)
        mascara_rojo2 = cv2.inRange(hsv, rojo_bajos2, rojo_altos2)

        # Juntar todas las mascaras
        mask = cv2.add(mascara_rojo1, mascara_rojo2)
        mask.tolist()

        mascaras.append(mask)

    for mascara in mascaras:
        mascara_media += mascara


mascara_media = mascara_media // len(mascaras)
print(mascara_media)
print("......................................")
condMenor = mascara_media < 200
condMayor = mascara_media >= 200
mascara_media[condMenor] = 0
mascara_media[condMayor] = 255
cv2.imshow("mask_media", mascara_media)
print(mascara_media)
cv2.waitKey()
