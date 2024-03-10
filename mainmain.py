from ultralytics import YOLO
model = YOLO("yolov8s.pt")
results = model.predict(source="img.jpg", show=True, verbose=False) # accepts all formats
# for r in results:
#     boxes = r.boxes  # Boxes object for bbox outputs
#     masks = r.masks  # Masks object for segment masks outputs
#     probs = r.probs  # Class probabilities for classification outputs
#     print(probs)

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