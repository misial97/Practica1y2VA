import cv2


_winSize = (30, 30)
_blockSize = (15, 15)
_blockStride = (5, 5)
_cellSize = (5, 5)
_nbins = 9

img = cv2.imread("Q:\Jorge\Documents\\va\Practica2\\test_reconocimiento\\01-00002.ppm")

img_recortada = cv2.resize(img, (30, 30), interpolation=cv2.INTER_LINEAR)

# Convertimos a escala de grises
#gray = cv2.cvtColor(img_recortada, cv2.COLOR_BGR2GRAY)

# Equalizamos hist. para mejorar contraste (detecta mas senyales usando esta imagen que la escala grises normal)
gray_eq = cv2.equalizeHist(img_recortada)

hog = cv2.HOGDescriptor(_winSize, blockSize, blockStride, cellSize, nbins)

hist = hog.compute(gray_eq)

print(hist)