import cv2
import numpy as np

cap=cv2.VideoCapture(0)
cap.open("http://192.168.1.2:8080/video")
imgTarget=cv2.imread("Refrence Images/Ref Image AR.jpg")
myVid=cv2.VideoCapture("Videos/Testimg.mp4")

success1,imgVideo=myVid.read()
hT,wT,cT=imgTarget.shape
imgVideo=cv2.resize(imgVideo,(wT,hT))

orb=cv2.ORB_create(nfeatures=2000)
kp1,des1=orb.detectAndCompute(imgTarget,None)
#imgTarget=cv2.drawKeypoints(imgTarget,kp1,None)

while True:
    success,imgWebcam=cap.read()
    kp2,des2=orb.detectAndCompute(imgWebcam,None)
    #imgWebcam = cv2.drawKeypoints(imgWebcam, kp2, None)

    bf=cv2.BFMatcher()
    matches=bf.knnMatch(des1,des2,k=2)
    good=[]

    for m,n in matches:
        if m.distance < 0.75* n.distance:
            good.append(m)
    print(len(good))

    imgFeatures=cv2.drawMatches(imgTarget,kp1,imgWebcam,kp2,good,None,flags=2)
    cv2.imshow("Image Features", imgFeatures)
    cv2.imshow("Image Target",imgTarget)
    cv2.imshow("Video Target",imgVideo)
    cv2.imshow("Webcam", imgWebcam)

    cv2.waitKey(0)