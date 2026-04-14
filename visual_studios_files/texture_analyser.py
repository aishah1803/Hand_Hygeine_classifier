# texture_analyser.py
# Analyses the texture of the hand region
# Uses OpenCV to detect how rough, smooth or consistent the surface looks

import cv2
import numpy as np

class TextureAnalyser:
    def __init__(self):
        pass

    def extract_region(self, image, coords):
        # Crop a region from the image using box coordinates
        x1, y1, x2, y2 = [int(c) for c in coords]
        region = image[y1:y2, x1:x2]
        return region

    def analyse_texture(self, image_path, hand_boxes, gel_boxes):
        # Load the image
        image = cv2.imread(image_path)

        # If no hand detected return early
        if not hand_boxes:
            return {
                "smoothness": 0,
                "consistency": 0,
                "roughness": 0,
                "gel_texture": None,
                "summary": "Could not analyse texture — no hand found"
            }

        # Get the best hand box
        best_hand = max(hand_boxes, key=lambda x: x["confidence"])
        hand_region = self.extract_region(image, best_hand["coords"])

        # Convert to greyscale for texture analysis
        # Greyscale is easier to analyse patterns and edges
        grey_hand = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)

        # ── Smoothness ──────────────────────────────────────────
        # Use Laplacian filter to detect edges
        # More edges = rougher surface
        # Fewer edges = smoother surface
        laplacian = cv2.Laplacian(grey_hand, cv2.CV_64F)
        roughness = float(np.var(laplacian))
        roughness = round(roughness, 2)

        # Convert roughness to smoothness score (0 to 100)
        # Higher roughness = lower smoothness
        smoothness = max(0, round(100 - min(roughness / 10, 100), 1))

        # ── Consistency ─────────────────────────────────────────
        # Standard deviation of pixel values
        # Low std dev = consistent colour = gel spread evenly
        # High std dev = patchy = uneven gel or no gel
        std_dev = float(np.std(grey_hand))
        consistency = max(0, round(100 - min(std_dev, 100), 1))

        # ── Gel texture ─────────────────────────────────────────
        gel_texture = None
        if gel_boxes:
            best_gel = max(gel_boxes, key=lambda x: x["confidence"])
            gel_region = self.extract_region(image, best_gel["coords"])

            if gel_region.size > 0:
                grey_gel = cv2.cvtColor(gel_region, cv2.COLOR_BGR2GRAY)

                gel_laplacian = cv2.Laplacian(grey_gel, cv2.CV_64F)
                gel_roughness = float(np.var(gel_laplacian))
                gel_std = float(np.std(grey_gel))
                gel_consistency = max(0, round(100 - min(gel_std, 100), 1))

                gel_texture = {
                    "roughness": round(gel_roughness, 2),
                    "consistency": gel_consistency
                }

        # ── Summary sentence ────────────────────────────────────
        if smoothness >= 70:
            smoothness_label = "smooth"
        elif smoothness >= 40:
            smoothness_label = "moderately textured"
        else:
            smoothness_label = "rough"

        if consistency >= 70:
            consistency_label = "evenly distributed"
        elif consistency >= 40:
            consistency_label = "partially distributed"
        else:
            consistency_label = "unevenly distributed"

        summary = (
            f"Hand surface is {smoothness_label} "
            f"(smoothness: {smoothness}/100). "
            f"Gel appears {consistency_label} "
            f"(consistency: {consistency}/100)."
        )

        return {
            "smoothness": smoothness,
            "consistency": consistency,
            "roughness": roughness,
            "gel_texture": gel_texture,
            "summary": summary
        } 