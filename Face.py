import cv2

cascPath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    ok,frame = video_capture.read()
    if not ok:
        break

    #* Transformar a imagem em gray para ler
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    #* Faces é uma lista onde fica armazenados os encontros de faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minSize=(60,60))


    for (x,y,w,h) in faces:

        #* Frame é oq foi capturado pela cam 
        cv2.rectangle(frame,(x,y), (x+w, y+h), (0,255,0),2)
        recorte = frame[y:y+w, x:x+w]

    cv2.imshow('Face detectada', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()