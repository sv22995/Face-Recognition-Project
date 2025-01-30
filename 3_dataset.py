import cv2

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
camera = cv2.VideoCapture(0)

ID = input("\n Enter your ID (like 1,2,3...) and press 'ENTER' : ")
print("\n Look at the camera, show your face with different angles, and wait until the data is collected... ")

count = 0

while True:
    ret , frame = camera.read()
    
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    Coordinates = detector.detectMultiScale(gray_img, 1.2, 5)
    
    for (x,y,w,h) in Coordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)     
        
        count = count + 1
        
        cv2.imwrite("dataset/ID."+ str(ID)+ "." + str(count) + ".jpg", gray_img[y:y+h,x:x+w])
        cv2.imshow("Dataset Collection", frame)
    
    if cv2.waitKey(1) == ord("q"):
        break
    elif count >= 50:
        break

camera.release()
cv2.destroyAllWindows()
print("\n Dataset collection successfully completed!")