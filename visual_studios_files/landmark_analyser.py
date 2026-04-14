# landmark_analyser.py
# Uses MediaPipe to detect hand landmarks (finger positions)
# Uses the newer MediaPipe API compatible with mediapipe 0.10+

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class LandmarkAnalyser:
    def __init__(self, model_path="models/hand_landmarker.task"):
        self.model_path = model_path

    def analyse_landmarks(self, image_path, hand_boxes):
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            return {
                "landmarks_found": False,
                "num_landmarks": 0,
                "fingers_detected": [],
                "hand_coverage": 0.0,
                "summary": "Could not load image"
            }

        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Set up MediaPipe hand landmarker using new API
        base_options = python.BaseOptions(
            model_asset_path=self.model_path
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )

        # Create the landmarker
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            # Convert image to MediaPipe format
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image_rgb
            )

            # Run detection
            detection_result = landmarker.detect(mp_image)

            # If no landmarks found
            if not detection_result.hand_landmarks:
                return {
                    "landmarks_found": False,
                    "num_landmarks": 0,
                    "fingers_detected": [],
                    "hand_coverage": 0.0,
                    "summary": "No hand landmarks detected"
                }

            # Get first hand landmarks
            landmark_list = detection_result.hand_landmarks[0]

            # Key landmark indices
            # 0=wrist, 4=thumb tip, 8=index tip
            # 12=middle tip, 16=ring tip, 20=pinky tip
            finger_tips = {
                "thumb":  landmark_list[4],
                "index":  landmark_list[8],
                "middle": landmark_list[12],
                "ring":   landmark_list[16],
                "pinky":  landmark_list[20]
            }

            finger_bases = {
                "index":  landmark_list[5],
                "middle": landmark_list[9],
                "ring":   landmark_list[13],
                "pinky":  landmark_list[17]
            }

            # Check which fingers are extended
            fingers_extended = []
            for finger in ["index", "middle", "ring", "pinky"]:
                tip_y = finger_tips[finger].y
                base_y = finger_bases[finger].y
                if tip_y < base_y:
                    fingers_extended.append(finger)

            # Check thumb
            if finger_tips["thumb"].x > landmark_list[3].x:
                fingers_extended.append("thumb")

            # Calculate hand spread
            x_coords = [lm.x for lm in landmark_list]
            y_coords = [lm.y for lm in landmark_list]
            x_spread = max(x_coords) - min(x_coords)
            y_spread = max(y_coords) - min(y_coords)
            hand_coverage = round((x_spread * y_spread) * 100, 1)

            # Build summary
            num_fingers = len(fingers_extended)
            if num_fingers >= 4:
                hand_position = "fully open hand"
            elif num_fingers >= 2:
                hand_position = "partially open hand"
            else:
                hand_position = "closed or fist hand"

            summary = (
                f"Detected {num_fingers} extended finger(s) — {hand_position}. "
                f"Hand spread: {hand_coverage}%."
            )

            return {
                "landmarks_found": True,
                "num_landmarks": len(landmark_list),
                "fingers_detected": fingers_extended,
                "num_fingers_extended": num_fingers,
                "hand_coverage": hand_coverage,
                "hand_position": hand_position,
                "summary": summary
            }

    def draw_landmarks(self, image_path, landmark_result):
        # Just return the image as is
        # Drawing landmarks requires extra setup with new API
        image = cv2.imread(image_path)
        return image 