import cv2

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
camera = cv2.VideoCapture(0)

while True:
    ret , frame = camera.read()

    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
 
    Coordinates = detector.detectMultiScale(gray_img,1.2,5)

    for(x,y,w,h) in Coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    cv2.putText(frame, "PRESS 'q' TO CLOSE THE WINDOW" , (260,25), cv2.FONT_ITALIC, 0.7 , (0,0,255), 1,  cv2.LINE_AA)
    cv2.imshow("Real time Face Detection", frame)
    
    if cv2.waitKey(1) == ord("q"):
        break

print("\n CONGRATS! YOUR FACE IS DETECTED SUCCESSFULLY")
camera.release()
cv2.destroyAllWindows()