import cv2
import matplotlib
import numpy as np

m = cv2.imread("/home/misial/Escritorio/stop.jpg")

cv2.imshow("m", m)
cv2.imshow("grey",cv2.cvtColor(m, cv2.COLOR_BGR2GRAY))
cv2.waitKey(0)

grey = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

mser = cv2.MSER_create()
regions, bboxes = mser.detectRegions(grey)
x1, y1, w, h = cv2.boundingRect(regions[200])
rect = (x1,y1,w,h)
rects = []
rects.append(rect)
cv2.rectangle(m, (x1, y1), (w+x1, h+y1), (0, 255, 0), -1);
cv2.imshow("aaaaa",m)
cv2.waitKey(0)

'''
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