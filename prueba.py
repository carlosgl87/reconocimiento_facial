import cv2
import numpy as np

rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('recognizer\\trainingData.yml')