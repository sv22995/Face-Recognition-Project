import cv2
camera = cv2.VideoCapture(0)

while True:
    ret , frame = camera.read()
  
    cv2.putText(frame, "PRESS 'q' TO CLOSE THE WINDOW" , (260,25), cv2.FONT_ITALIC, 0.7 , (0,0,255), 1,  cv2.LINE_AA)
    cv2.imshow("video read", frame)
    
    if cv2.waitKey(1) == ord("q"):
        break

print("CONGRATS! YOUR WEBCAM IS ABLE TO READ YOUR VIDEO!")
camera.release()
cv2.destroyAllWindows()