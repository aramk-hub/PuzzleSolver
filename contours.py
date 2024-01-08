# import cv2 as cv
# import numpy as np

# # Read in image of puzzle pieces
# image = cv.imread("./Pieces/IMG_8340.jpg")
  
# # Grayscale 
# gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) 
  
# # Find Canny edges 
# edged = cv.Canny(gray, 30, 200) 

# contours, hierarchy = cv.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# cv.imshow("Contours", edged)

# print("Number of Contours found = " + str(len(contours))) 
  
# # Draw all contours 
# # -1 signifies drawing all contours 
# cv.drawContours(image, contours, -1, (0, 255, 0), 3) 

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('./Pieces/IMG_8340.jpg', cv.IMREAD_GRAYSCALE)

assert img is not None, "file could not be read, check with os.path.exists()"

ret, thresh = cv.threshold(img, 127, 255, 0)

contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

cv.drawContours(img, contours, -1, (0,255,0), 3)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(thresh,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

# continuing
