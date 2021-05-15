import cv2
import numpy as np

cap=cv2.VideoCapture(0)
#cap.open("http://192.168.1.2:8080/video")
imgTarget=cv2.imread("Refrence Images/Ref Image AR.jpg")
myVid=cv2.VideoCapture("Videos/Testimg.mp4")

detection=False
frameCounter=0


success1,imgVideo=myVid.read()
hT,wT,cT=imgTarget.shape
imgVideo=cv2.resize(imgVideo,(wT,hT))

orb=cv2.ORB_create(nfeatures=2000)
kp1,des1=orb.detectAndCompute(imgTarget,None)
#imgTarget=cv2.drawKeypoints(imgTarget,kp1,None)

def stackImages(imgArray,scale,lables=[]):
    sizeW=imgArray[0][0].shape[1]
    sizeH=imgArray[0][0].shape[0]
    rows=len(imgArray)
    cols=len(imgArray[0])
    rowsAvailable=isinstance(imgArray[0],list)
    if rowsAvailable:
        for x in range(0,rows):
            for y in range(0,cols):
                imgArray[x][y]=cv2.resize(imgArray[x][y],(sizeW,sizeH),None,scale,scale)
                if len(imgArray[x][y].shape)==2:imgArray[x][y]=cv2.cvtColor(imgArray[x][y],cv2.COLOR_GRAY2BGR)
        imageBlank=np.zeros((sizeH,sizeW,2),np.uint8)
        hor=[imageBlank]*rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x]=np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver=np.vstack(hor)
        ver_con=np.concatenate(hor)
    else:
        for x in range(0,rows):
            imgArray[x] = cv2.resize(imgArray[x], (sizeW, sizeH), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor=np.hstack(imgArray)
        hor_con=np.concatenate(imgArray)
        ver=hor

    return ver



while True:
    success,imgWebcam=cap.read()
    imgAug = imgWebcam.copy()

    kp2,des2=orb.detectAndCompute(imgWebcam,None)
    #imgWebcam = cv2.drawKeypoints(imgWebcam, kp2, None)

    if detection==False:
        myVid.set(cv2.CAP_PROP_POS_FRAMES,0)
        frameCounter=0
    else:
        if frameCounter==myVid.get(cv2.CAP_PROP_FRAME_COUNT):
            myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        success,imgVideo=myVid.read()
        imgVideo = cv2.resize(imgVideo, (wT, hT))



    bf=cv2.BFMatcher()
    matches=bf.knnMatch(des1,des2,k=2)
    good=[]

    for m,n in matches:
        if m.distance < 0.75* n.distance:
            good.append(m)
    print(len(good))

    imgFeatures=cv2.drawMatches(imgTarget,kp1,imgWebcam,kp2,good,None,flags=2)

    if len(good)>20:
        detection=True
        scrpts=np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dstpts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        matrix,mask=cv2.findHomography(scrpts,dstpts,cv2.RANSAC,5)
        print(matrix)

        pts=np.float32([[0,0],[0,hT],[wT,hT],[wT,0]]).reshape(-1,1,2)
        dst=cv2.perspectiveTransform(pts,matrix)
        img2=cv2.polylines(imgWebcam,[np.int32(dst)],True,(255,0,255),3)

        imgWarp=cv2.warpPerspective(imgVideo,matrix,(imgWebcam.shape[1],imgWebcam.shape[0]))

        maskNew=np.zeros((imgWebcam.shape[0],imgWebcam.shape[1]),np.uint8)
        cv2.fillPoly(maskNew,[np.int32(dst)],(255,255,255))
        maskInv=cv2.bitwise_not(maskNew)
        imgAug=cv2.bitwise_and(imgAug,imgAug,mask=maskInv)
        imgAug=cv2.bitwise_or(imgWarp,imgAug)
        #imgAug = cv2.resize(imgAug, (0, 0), fx=0.5, fy=0.5)

        #imgStacked=stackImages(([imgaug,imgFeatures],[imgWarp,imgAug]),0.5)

    imgWebcam=cv2.resize(imgWebcam, (0, 0), fx=0.5, fy=0.5)
    imgFeatures = cv2.resize(imgFeatures, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("Live VDO",imgWebcam)
    cv2.imshow("Features Compare", imgFeatures)

    #cv2.imshow("AR VDO", imgAug)
    cv2.waitKey(1)
    frameCounter+=1




    #imgWarp = cv2.resize(imgWarp, (wT, hT))
    #img2 = cv2.resize(img2, (wT, hT))
    #imgFeatures = cv2.resize(imgFeatures, (wT, hT))
    # cv2.imshow("Img Mask", maskNew)
    # cv2.imshow("Img Inv Mask", maskInv)
    # # cv2.imshow("Img Warp", imgWarp)
    # # cv2.imshow("Image 2", img2)
    # # cv2.imshow("Image Features", imgFeatures)
    # # cv2.imshow("Image Target",imgTarget)
    # # cv2.imshow("Video Target",imgVideo)
    # cv2.imshow("Webcam Img Masked", imgAug)
    # cv2.imshow("Webcam", imgWebcam)