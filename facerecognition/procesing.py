import cv2
import numpy as np

# The function cv2.imread() is used to read an image.
img_grayscale = cv2.imread('images/art.jpeg', 0)

# The function cv2.imshow() is used to display an image in a window.

cv2.imshow('preview', img_grayscale)


#HaarCascade classifier is used for detecting frontal faces from the file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                                     
# detectMultiScale applies the classifier on a grayscale version of the input image                                      
faces = face_cascade.detectMultiScale(img_grayscale, 1.3, 5)                                    
 
#cv2.rectangle draws rectangles around the detected faces                                    
for (x,y,w,h) in faces:
    cv2.rectangle(img_grayscale,(x,y),(x+w,y+h),(255,0,0),2)
           
                                     
#prints the number of faces found.                                     
print("Faces found ", len(faces))

cv2.imshow('Image',img_grayscale)                                     

# waitKey() waits for a key press to close the window and 0 specifies indefinite loop
cv2.waitKey(0)

# cv2.destroyAllWindows() simply destroys all the windows we created.
cv2.destroyAllWindows()

# The function cv2.imwrite() is used to write an image.
cv2.imwrite('grayscale.jpg', img_grayscale)
