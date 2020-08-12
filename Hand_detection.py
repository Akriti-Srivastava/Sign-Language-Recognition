import cv2
import os


dirFace = 'cropped_hand'

# Create if there is no cropped face directory
if not os.path.exists(dirFace):
    os.mkdir(dirFace)
    print("Directory " , dirFace ,  " Created ")
else:
    print("Directory " , dirFace ,  " is found.")

cap = cv2.VideoCapture(0)
hand_cascade = cv2.CascadeClassifier('haarcascade/fist.xml')

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hands = hand_cascade.detectMultiScale(gray, 1.5, 2)
    for (x, y, w, h) in hands:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Welcome', frame)

    key = cv2.waitKey(1) & 0xFF
    # quit and break out of the loop
    if key == ord('q'):
        break
    #  capture the image of the cropped hand
    if key == ord('c'):
        sub_hand = frame[y:y + h, x:x + w]
        FaceFileName = "cropped_hand/hand_" + str(y + x) + ".jpg"
        cv2.imwrite(FaceFileName, sub_hand)

cap.release()
cv2.destroyAllWindows()
