import cv2
import numpy as np
import httplib, urllib, base64, json

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
rec = cv2.createLBPHFaceRecognizer()
rec.load('recognizer\\trainingData.yml')


subscription_key = '89d05f558b1a424893ffb65fe4a53952'
uri_base = 'westcentralus.api.cognitive.microsoft.com'
headers = {
    'Content-Type': 'application/json',
    'Ocp-Apim-Subscription-Key': subscription_key,
}
params = urllib.urlencode({
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise',
})

def llamar_api(body):
    conn = httplib.HTTPSConnection('westcentralus.api.cognitive.microsoft.com')
    conn.request("POST", "/face/v1.0/detect?%s" % params, body, headers)
    response = conn.getresponse()
    data = response.read()
    # 'data' contains the JSON data. The following formats the JSON data for display.
    parsed = json.loads(data)
    conn.close()
    return parsed


id = 0
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,2,1,0,2)
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf = rec.predict(gray[y:y+h,x:x+w])
        if id==1:
            id = 'Carlos Gamero'
        if id==2:
            id = 'Miguel Paredes'
        if id==3:
            id = 'Jose Naranjo'
        if id==4:
            id = 'Claudia'
        cv2.cv.PutText(cv2.cv.fromarray(img),str(id), (x,y+h),font,255)
        #print llamar_api(img[y:y + h, x:x + w])
        #cv2.imwrite('dataSet/Prueba.jpg', img[y:y + h, x:x + w])

    cv2.imshow("Face",img)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()