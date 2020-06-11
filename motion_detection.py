import cv2, time
first_frame = None
video = cv2.VideoCapture('./Test1.mp4')
while True:
    time.sleep(0.1)
    check, frame = video.read()
    # frame = cv2.resize(frame,(int(frame.shape[1]*3),int(frame.shape[0]*3)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)
    if first_frame is None:
        first_frame = gray
        continue
    delta_frame = cv2.absdiff(first_frame,gray)
    thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_delta = cv2.dilate(thresh_delta,None,iterations=0)
    (cnts,_) = cv2.findContours(thresh_delta.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in cnts:
        if(cv2.contourArea(contour)<1000):
            continue
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),3)
    
    cv2.imshow("Jay",frame)
    cv2.imshow("Jay1",gray)
    cv2.imshow("Jay2",delta_frame)
    cv2.imshow("Jay3",thresh_delta)
    key = cv2.waitKey(1)
    if key==ord("q"):
        break
video.release()
cv2.destroyAllWindows()