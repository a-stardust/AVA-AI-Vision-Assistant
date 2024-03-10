import cv2
from time import sleep
import keyboard 
from ultralytics import YOLO
model = YOLO("yolov8s.pt")



cap = cv2.VideoCapture(1)  # Webcam capture

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    cv2.imwrite('frame.jpg', frame)
    results = model.predict(source="frame.jpg", show=False, verbose=False, conf=0.5)
    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q') or keyboard.is_pressed('q'):
        break
    # Wait for 5 seconds
    sleep(5)
    for r in results:
        log_string = ""
        probs = r.probs
        boxes = r.boxes
        if len(r) == 0:
            if probs is not None:
                print(log_string)
            else :
                print(f"{log_string}(no detections), ")
        if probs is not None:
            log_string += f"{', '.join(f'{r.names[j]} {probs.data[j]:.2f}' for j in probs.top5)}, "
        if boxes:
            for c in boxes.cls.unique():
                n = (boxes.cls == c).sum()  # detections per class
                log_string += f"{n} {r.names[int(c)]}{'s' * (n > 1)}, "
        print(log_string)
cap.release()
cv2.destroyAllWindows()



