import cv2
import dlib

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    print(frame.shape) #720, 1280, 3
    if(len(faces) == 0):
        x1 = 600
        x2 = 800
        y1 = 500
        y2 = 500
        cv2.rectangle(frame, (x1, y2-35), (x2, y2), (0,255,0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, "WITH MASK", (x1 + 6, y2 - 6), font, 1.0, (255,255,255),1)
        

    #cv2.rectangle(frame, 
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        landmarks = predictor(image=gray, box=face)

        for n in range(0,68):
            x = landmarks.part(38).x
            y = landmarks.part(38).y
            
            #cv2.circle(img = frame, center = (x,y), radius = 3, color=(0,255,0), thickness = -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2) #red rectangle
        cv2.rectangle(frame, (x1, y2-35), (x2, y2), (0,0,255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, "NO MASK", (x1 + 6, y2 - 6), font, 1.0, (255,255,255),1)
        
    cv2.imshow(winname="face", mat=frame)

    if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()



