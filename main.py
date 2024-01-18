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

from weakref import ref
import numpy as np
import cv2 as cv
import scipy
import scipy.ndimage
from skimage.util import img_as_ubyte
from matplotlib import pyplot as plt

ref_image = cv.imread('solution.jpg', cv.IMREAD_GRAYSCALE)
ref_image = np.asarray(ref_image, dtype="uint8") / 255.0
ref_copy = ref_image

img = cv.imread('./Pieces/IMG_8340.jpg', cv.IMREAD_GRAYSCALE)
img = np.asarray(img, dtype="uint8") / 255.0

h, w = ref_image.shape
out_img = np.zeros((h, w))

bin_img = np.zeros(img.shape)
bin_img[img > 0.06] = 1.0
opened_img=scipy.ndimage.morphology.binary_opening(bin_img, 
                 structure=np.ones((3,3)), iterations=1, output=None, origin=0)
opened_img=scipy.ndimage.morphology.binary_closing(opened_img, 
             structure=np.ones((3,3)), iterations=1, output=None, origin=0)

vis_img = img_as_ubyte(opened_img)
im2, contours, hierarchy = cv.findContours(vis_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


cv.drawContours(img, contours, -1, (255,255,0), 2)
plt.imshow(im2,cmap=plt.cm.gray)
plt.title("Pieces")
plt.axis('off')
plt.show()

# Create the array of pieces by going through contours
pieces = list()
for i in range(0, len(contours)):
    curr_contours = np.asarray(contours[i])
    x,y,w,h = cv.boundingRect(curr_contours)
    cv.rectangle(im2, (x,y), (x+w, y+h), (255, 255, 0), 2)
    cropped = img[y:y+h, x:x+w]
    pieces.append(cropped)

plt.imshow(im2,cmap=plt.cm.gray)
plt.axis('off')
plt.show()

# Create SIFT Detector 
sift = cv.SIFT_create()

ref_copy = img_as_ubyte(ref_copy)
kp2, desc2 = sift.detectAndCompute(ref_copy, None)

bf = cv.BFMatcher()

# Begin Matching Sequence


# continuing
