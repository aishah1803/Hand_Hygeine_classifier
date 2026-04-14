# colour_analyser.py
# Analyses the colour of the hand and gel regions
# Uses OpenCV to extract colour information

import cv2
import numpy as np

class ColourAnalyser:
    def __init__(self):
        pass

    def extract_region(self, image, coords):
        # Crop a region from the image using box coordinates
        x1, y1, x2, y2 = [int(c) for c in coords]
        region = image[y1:y2, x1:x2]
        return region

    def analyse_colour(self, image_path, hand_boxes, gel_boxes):
        # Load the image
        image = cv2.imread(image_path)

        # If no hand detected return early
        if not hand_boxes:
            return {
                "hand_colour": None,
                "gel_colour": None,
                "brightness": 0,
                "gel_spread": "No hand detected",
                "summary": "Could not analyse colour — no hand found"
            }

        # Get the best hand box
        best_hand = max(hand_boxes, key=lambda x: x["confidence"])
        hand_region = self.extract_region(image, best_hand["coords"])

        # Convert hand region to HSV colour space
        # HSV = Hue Saturation Value, easier to analyse colour than RGB
        hand_hsv = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)

        # Get average colour values of the hand
        avg_hue = float(np.mean(hand_hsv[:, :, 0]))
        avg_saturation = float(np.mean(hand_hsv[:, :, 1]))
        avg_brightness = float(np.mean(hand_hsv[:, :, 2]))

        # Get average BGR colour of hand (Blue Green Red)
        avg_blue = float(np.mean(hand_region[:, :, 0]))
        avg_green = float(np.mean(hand_region[:, :, 1]))
        avg_red = float(np.mean(hand_region[:, :, 2]))

        hand_colour = {
            "hue": round(avg_hue, 1),
            "saturation": round(avg_saturation, 1),
            "brightness": round(avg_brightness, 1),
            "blue": round(avg_blue, 1),
            "green": round(avg_green, 1),
            "red": round(avg_red, 1)
        }

        # Analyse gel colour if gel detected
        gel_colour = None
        gel_spread = "No gel detected"

        if gel_boxes:
            # Get the gel region
            best_gel = max(gel_boxes, key=lambda x: x["confidence"])
            gel_region = self.extract_region(image, best_gel["coords"])

            if gel_region.size > 0:
                gel_hsv = cv2.cvtColor(gel_region, cv2.COLOR_BGR2HSV)

                gel_avg_hue = float(np.mean(gel_hsv[:, :, 0]))
                gel_avg_saturation = float(np.mean(gel_hsv[:, :, 1]))
                gel_avg_brightness = float(np.mean(gel_hsv[:, :, 2]))

                gel_colour = {
                    "hue": round(gel_avg_hue, 1),
                    "saturation": round(gel_avg_saturation, 1),
                    "brightness": round(gel_avg_brightness, 1)
                }

                # Compare brightness of hand vs gel
                # If gel region is significantly different brightness
                # it means gel is visibly present and spread
                brightness_diff = abs(avg_brightness - gel_avg_brightness)

                if brightness_diff > 40:
                    gel_spread = "Gel clearly visible and well spread"
                elif brightness_diff > 20:
                    gel_spread = "Gel partially visible"
                else:
                    gel_spread = "Gel spread evenly across hand"

        # Overall colour summary
        if avg_brightness > 150:
            brightness_label = "well lit"
        elif avg_brightness > 80:
            brightness_label = "normally lit"
        else:
            brightness_label = "poorly lit"

        summary = (
            f"Hand is {brightness_label} "
            f"(brightness: {round(avg_brightness, 1)}). "
            f"Gel assessment: {gel_spread}."
        )

        return {
            "hand_colour": hand_colour,
            "gel_colour": gel_colour,
            "brightness": avg_brightness,
            "gel_spread": gel_spread,
            "summary": summary
        } 