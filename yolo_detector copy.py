import cv2
from time import sleep
import keyboard
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path="yolov8s.pt"):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(1)

    def detect_objects(self, confidence_threshold=0.5):
        log_string = ""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            cv2.imwrite('frame.jpg', frame)
            results = self.model.predict(source="frame.jpg", show=False, verbose=False, conf=confidence_threshold)

            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q') or keyboard.is_pressed('q'):
                break

            for r in results:
                probs = r.probs
                boxes = r.boxes
                if len(r) == 0:
                    if probs is not None:
                        print(log_string)
                    else:
                        print(f"{log_string}(no detections), ")
                if probs is not None:
                    log_string += f"{', '.join(f'{r.names[j]} {probs.data[j]:.2f}' for j in probs.top5)}, "
                if boxes:
                    for c in boxes.cls.unique():
                        n = (boxes.cls == c).sum()  # detections per class
                        log_string += f"{n} {r.names[int(c)]}{'s' * (n > 1)}, "
        return log_string

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

# Example usage in another script:
# from yolo_detector import YoloDetector
# detector = YoloDetector()
# log_string = detector.detect_objects()
# print(log_string)
# detector.release()
