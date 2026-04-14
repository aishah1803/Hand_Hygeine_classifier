# detector.py
# This file loads your YOLO11 model and detects hands and gel in an image

from ultralytics import YOLO
import cv2

class HandGelDetector:
    def __init__(self, model_path):
        # Load your trained YOLO11 model
        # model_path = the location of your best.pt file
        self.model = YOLO(model_path)
        print("Model loaded successfully from:", model_path)

    def detect(self, image_path):
        # Run YOLO11 on the image
        results = self.model.predict(
            source=image_path,
            imgsz=640,
            conf=0.3,        # confidence threshold - only show detections above 30%
            verbose=False    # dont print extra info
        )

        result = results[0]  # get first image result

        # These will store what we find
        hand_boxes = []
        gel_boxes = []

        # Loop through every detection
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls)          # 0 = Hand, 1 = Gel
                confidence = float(box.conf)      # how confident YOLO is
                coords = box.xyxy[0].tolist()     # box coordinates [x1, y1, x2, y2]

                if class_id == 1:  # Hand detected
                    hand_boxes.append({
                        "confidence": confidence,
                        "coords": coords
                    })
                elif class_id == 0:  # Gel detected
                    gel_boxes.append({
                        "confidence": confidence,
                        "coords": coords
                    })

        return hand_boxes, gel_boxes, result

    def draw_boxes(self, image_path, hand_boxes, gel_boxes):
        # Load the image
        image = cv2.imread(image_path)

        # Draw green boxes around hands
        for hand in hand_boxes:
            x1, y1, x2, y2 = [int(c) for c in hand["coords"]]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"Hand {hand['confidence']:.0%}",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 0), 2)

        # Draw blue boxes around gel
        for gel in gel_boxes:
            x1, y1, x2, y2 = [int(c) for c in gel["coords"]]
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, f"Gel {gel['confidence']:.0%}",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 0, 0), 2)

        return image 