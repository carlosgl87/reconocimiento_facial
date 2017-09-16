# Face Recognition / Reconocimiento Facial
Python program for facial recognition with numpy, pillow and OpenCV

To use the program, first you have to run the file: **dataset_creator.py**  
***dataset_creator*** activate the camera and take 100 photos from a face. You have to indicate the ID of the face that you are taking pictures

Then, to train the facial recognition algorithm you have to run the file **trainer**
***trainer*** create a file named *trainingData.yml* in the recognizer folder. With this file the program is able to recognize a face

Finally, with file **detector.py** activates the camera and detect in real time the faces and the ID. In the code you can change the ID with a name, and this name will be displayed in real time in the video.
