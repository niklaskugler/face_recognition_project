import cv2
from faceRecognition import FaceRecognition
import playsound

def userFeedback_sound(counter, name):
    # Zugang gewährt oder nicht gewährt nach gewisser Anzahl geprüfter Frames
    if (counter == 20): 
            # play sound file
            playsound.playsound('access_granted.mp3', True)
            counter = 0
    elif (counter == -20):
        # play sound file
        playsound.playsound('access_denied.mp3', True)
        counter = 0
    elif name == "NiklasKugler": 
        counter = counter+1
    elif name == "Unknown": 
        counter = counter-1
    else: 
        counter = 0
    return counter

def main(): 
    # Codiere Gesichter aus Samples aus Ordner images
    sfr = FaceRecognition()
    sfr.load_encoding_images("images/")

    # Kamera laden
    cap = cv2.VideoCapture(-1)

    counter = 1

    while True:
        ret, frame = cap.read()

        # detektiere Gescichter mit detect_known_faces 
        face_locations, face_names = sfr.detect_known_faces(frame)

        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            # Farbe in BGR
            if name == "NiklasKugler":
                cv2.putText(frame, "Hallo Niklas!",(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 200, 50), 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 200, 50), 4)
                
            else: 
                cv2.putText(frame, "Unbekannte Person. Kein Zutritt!",(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (50, 50, 200), 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 200), 4)

            ## Sound Acceess to the shuttle
            # offene todos erweitern das sysetme
            # print(counter name)
            counter = userFeedback_sound(counter, name)

        cv2.imshow("Frame", frame)

        # Beende Programm mit q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__": 
    main()
