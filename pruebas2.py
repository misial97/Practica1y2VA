import cv2 as cv
import numpy as np

img = cv.imread("/home/misial/Descargas/train/00004.ppm")

_delta = 4
_min_area = 60
_max_area = 10000
_max_variation = 0.15
_min_diversity = 0.2
_max_evolution = 200
_area_threshold = 1.01
_min_margin = 0.003
_edge_blur_size = 5

mser = cv.MSER_create(_delta,_min_area, _max_area, _max_variation,_min_diversity,_max_evolution, _area_threshold, _min_margin, _edge_blur_size)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
vis = img.copy()

regions, _ = mser.detectRegions(gray)
hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]
cv.polylines(vis, hulls, 1, (0, 255, 0))
print(hulls)
cv.imshow('img', vis)
cv.waitKey()

'''import cv2
import matplotlib
import numpy as np

m = cv2.imread("/home/misial/Escritorio/stop.jpg")

cv2.imshow("m", m)
cv2.imshow("grey",cv2.cvtColor(m, cv2.COLOR_BGR2GRAY))
cv2.waitKey(0)

grey = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

mser = cv2.MSER_create()
regions, bboxes = mser.detectRegions(grey)
x1, y1, w, h = cv2.boundingRect(regions[1000])
print(regions)
rect = (x1,y1,w,h)
rects = []
rects.append(rect)
cv2.rectangle(m, (x1, y1), (w+x1, h+y1), (0, 255, 0), -1);
cv2.imshow("aaaaa",m)
cv2.waitKey(0)


print(m.shape)

b = m[:,:,0]*0.11
g = m[:,:,1]*0.59
r = m[:,:,2]*0.11

t = np.uint8(b+g+r)
mser = cv2.MSER_create()
regions, _ = mser.detectRegions(t)

hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

cv2.polylines(t, hulls, 1, (0, 255, 0))

cv2.imshow('img', t)

cv2.waitKey(0)

"""
uint8 -- los valores estaran en 0 y 255
float32 --- los valores estaran entre 0 y 1
"""
cv2.imshow("alg", t)
cv2.waitKey()
"""""
for i in range(0,281):
    for j in range(0,500):
        if((m[i,j,0]<=m[i,j,1])and(m[i,j,0]<=m[i,j,2])):
            m[i, j, 0] = 69
            m[i, j, 1] = 253
            m[i, j, 2] = 39

cv2.imshow("ej2_1", np.uint8(m))
cv2.waitKey()
"""

cond = (m[:,:,0] < m[:,:,1]) | (m[:,:,0] < m[:,:,2])
print(cond)


b1 = m[:,:,0]
g1 = m[:,:,1]
r1 = m[:,:,2]

r1[cond] = 255
g1[cond] = 255
b1[cond] = 255

m[:,:,2]= r1
m[:,:,1]= g1
m[:,:,0]= b1


cv2.imshow("ej2", np.uint8(m))
cv2.waitKey()


I = cv2.imread("/home/misial/Escritorio/avatar.jpg", 0)
channels = [0]
histSize = [256]
range_hist = [0, 256]
hist0 = cv2.calcHist([I], channels, None, histSize, range_hist)

minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist0)
altura = 255
hist0 = (altura*hist0) // maxVal # División entera es //
histImage = np.zeros((altura,256), np.uint8)
for i in range(0, 256):
    cv2.line(histImage, (i, 255), (i, altura-hist0[i]), 255)
cv2.imshow("histo",histImage)
cv2.waitKey()
'''