import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer.read("trainer/trainer.yml")

camera = cv2.VideoCapture(0)

ID = 0
Name = ['None', "Mr. Narendra Modi", "Bill Gates"]

while True:
    
    ret , frame = camera.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    Coordinates = detector.detectMultiScale(gray_img,1.3,5)
    
    for(x,y,w,h) in Coordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        ID , confidence = recognizer.predict(gray_img[y:y+h,x:x+w])
    
        if(confidence < 100):
            ID = Name[ID]
            confidence = "  {0}%".format(round(100 - confidence))
        else: 
            n = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(frame, str(ID) , (x+10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    cv2.putText(frame, "PRESS 'q' TO CLOSE THE WINDOW" , (260,25), cv2.FONT_ITALIC, 0.7 , (0,0,255), 1,  cv2.LINE_AA)
    cv2.imshow("REAL TIME FACE RECOGNITION", frame) 

    if cv2.waitKey(1) == ord("q"):
        break

print("\n CONGRATS! YOUR FACE HAS BEEN RECOGNIZED SUCCESSFULLY")
camera.release()
cv2.destroyAllWindows()