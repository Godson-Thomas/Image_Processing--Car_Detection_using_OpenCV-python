<img src="https://github.com/Godson-Thomas/Image_Processing--Car_Detection_using_OpenCV-python/blob/master/Images%20And%20Videos/1.png" width="250"> <br><br>

# COMPUTER VISION
Computer Vision, often abbreviated as CV, is defined as a field of study that seeks to develop techniques to help computers “see” and understand the content of digital images such as photographs and videos.Moreover Computer vision focuses on replicating parts of the complexity of the human vision system and enabling computers to identify and process objects in images and videos in the same way that humans do. <br>**Image processing**  is a method to perform some operations on an image, in order to get an enhanced image or to extract some useful information from it.<br>
Here we are going to detect cars from a video by using **HAAR CASCADE** classifier.
# HAAR CASCADE
Haar Cascade is a machine learning object detection algorithm used to identify objects in an image or video and based on the concept of ​​ features proposed by Paul Viola and Michael Jones in their paper "Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001.<br>
OpenCV already contains many pre-trained classifiers for face, eyes, smile etc..So we will be using one of the pre-trained classifier here.
# Library
OpenCV (Open Source Computer Vision) is a library of programming functions mainly aimed at real-time computer vision. In short it is a library used for Image Processing. It is mainly used to do all the operation related to Images.We will be using this library.
### Note : ***Video is basically a sequence of moving images."Persistance of vision" (speciality of our eyes)  plays a major role in receiving moving images.*** 
# Steps
## Installation :


1. We will be using Jupyter Notebook for writing the code.Make sure you have Jupyter Notebook installed.<br><br>
2. Lauch your Jupyter Notebook<br><br>
3. Now we have to install the OpenCV library.Type the code in the cell of Jupyter Notebook and run it.
```
pip install opencv-python
```
<br>
<img src="https://github.com/Godson-Thomas/Image_Processing---Facial-Detection-Using-OpenCV/blob/master/Images/2.png" width="500" height=75>  <br><br> 

4. - Download the ***Haar Cascade Classifier***. [click here](https://raw.githubusercontent.com/Godson-Thomas/Image_Processing--Car_Detection_using_OpenCV-python/master/cars.xml)<br>
-    Download a sample video.[click here](https://github.com/Godson-Thomas/Image_Processing--Car_Detection_using_OpenCV-python/blob/master/Images%20And%20Videos/1video.avi)

 ## Code :
 ### Type the codes in the cell and run it.<br><br>
5. Import the OpenCV library and time module.
```
import cv2
import time
```
6. Load your video which is to be detected to a variable using this code.
```
video=cv2.VideoCapture("/Videos/video.avi")
                                 # Video location
```
7. Now read the Haar Cascade classifier.
```
car_cascade=cv2.CascadeClassifier("/Videos/cars.xml")
                                 # Classifier Location                 
```
8. Since a video is a sequence of images, we have to go through each and every frame using a loop.We will use the haar cascade feature in every frame.Then we'll try to detect car in every frame.<br><br>
9. When a car is detected,we'll draw a rectangle to indicate it.
```
a=1

while True:
    
    
    
    
    check,img = video.read(0)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(
        gray,

        scaleFactor=1.07,
        minNeighbors=3,                 # Iterating through each frame
        minSize=(30, 30)
        )

    

    for (x ,y ,w ,h) in cars:
        cv2.rectangle(img ,(x ,y) ,( x +w , y +h) ,(255 ,0 ,0) ,2)

    
    cv2.imshow("video" ,img)
                                                             #Drawing Rectangle
    k=cv2.waitKey(30)
    a=a+1
    if k==ord('q'):     # press 'q' to quit
        break

video.release()
cv2.destroyAllWindows()

```
<br>

10. Every frame is displayed on a window using :
```
 cv2.imshow("video" ,img)
 ```
 ### Note :
 Make sure that you destroy all the windows you opened.
 ```
 video.release()
cv2.destroyAllWindows()

```
### Full Code :
[Click here](https://github.com/Godson-Thomas/Image_Processing--Car_Detection_using_OpenCV-python/blob/master/Detection.ipynb)