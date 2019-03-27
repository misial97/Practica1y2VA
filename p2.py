import cv2

#-----Reading the image-----------------------------------------------------
img = cv2.imread("/home/misial/Escritorio/00323.ppm", 1)
cv2.imshow("img",img)
cv2.waitKey(0)

#-----Converting image to LAB Color model-----------------------------------
lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.waitKey(0)


#-----Splitting the LAB image to different channels-------------------------
l, a, b = cv2.split(lab)
cv2.imshow('l_channel', l)
cv2.waitKey(0)

cv2.imshow('a_channel', a)
cv2.waitKey(0)

mser = cv2.MSER_create()
regions, _ = mser.detectRegions(a)

hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

cv2.polylines(a, hulls, 1, (0, 255, 0))

cv2.imshow('REGIONES', a)

cv2.waitKey(0)


cv2.imshow('b_channel', b)
cv2.waitKey(0)

#-----Applying CLAHE to L-channel-------------------------------------------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
cv2.imshow('CLAHE output', cl)
cv2.waitKey(0)


#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
limg = cv2.merge((cl,a,b))
cv2.imshow('limg', limg)
cv2.waitKey(0)


#-----Converting image from LAB Color model to RGB model--------------------
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
cv2.imshow('final', final)
cv2.waitKey(0)

final2 = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
cv2.imshow("final2", final2)
cv2.waitKey()
#_____END_____#