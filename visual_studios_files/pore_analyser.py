# pore_analyser.py
# Analyses pore visibility and skin texture detail
# Uses OpenCV filters to detect fine surface details

import cv2
import numpy as np

class PoreAnalyser:
    def __init__(self):
        pass

    def extract_region(self, image, coords):
        # Crop a region from the image using box coordinates
        x1, y1, x2, y2 = [int(c) for c in coords]
        region = image[y1:y2, x1:x2]
        return region

    def analyse_pores(self, image_path, hand_boxes):
        # Load image
        image = cv2.imread(image_path)

        if image is None:
            return {
                "pores_detected": False,
                "pore_count": 0,
                "pore_density": 0.0,
                "pore_size_avg": 0.0,
                "gel_coverage_effect": "unknown",
                "summary": "Could not load image"
            }

        # If no hand detected return early
        if not hand_boxes:
            return {
                "pores_detected": False,
                "pore_count": 0,
                "pore_density": 0.0,
                "pore_size_avg": 0.0,
                "gel_coverage_effect": "unknown",
                "summary": "No hand detected for pore analysis"
            }

        # Get best hand region
        best_hand = max(hand_boxes, key=lambda x: x["confidence"])
        hand_region = self.extract_region(image, best_hand["coords"])

        if hand_region.size == 0:
            return {
                "pores_detected": False,
                "pore_count": 0,
                "pore_density": 0.0,
                "pore_size_avg": 0.0,
                "gel_coverage_effect": "unknown",
                "summary": "Hand region too small to analyse"
            }

        # ── Step 1: Convert to greyscale ────────────────────────
        grey = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)

        # ── Step 2: Enhance fine details ────────────────────────
        # CLAHE = Contrast Limited Adaptive Histogram Equalisation
        # Makes small details like pores more visible
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(grey)

        # ── Step 3: Blur slightly to remove noise ───────────────
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # ── Step 4: Detect small dark spots (pores) ─────────────
        # Pores appear as small dark circles on skin
        # We use SimpleBlobDetector to find them
        params = cv2.SimpleBlobDetector_Params()

        # Filter by colour — pores are darker than skin
        params.filterByColor = True
        params.blobColor = 0  # 0 = dark blobs

        # Filter by size — pores are small
        params.filterByArea = True
        params.minArea = 2
        params.maxArea = 150

        # Filter by shape — pores are roughly circular
        params.filterByCircularity = True
        params.minCircularity = 0.3

        # Filter by convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(blurred)

        pore_count = len(keypoints)

        # ── Step 5: Calculate pore density ──────────────────────
        # Pore density = pores per 1000 pixels of hand area
        hand_area = hand_region.shape[0] * hand_region.shape[1]
        pore_density = round((pore_count / max(hand_area, 1)) * 1000, 4)

        # ── Step 6: Calculate average pore size ─────────────────
        if keypoints:
            sizes = [kp.size for kp in keypoints]
            avg_size = round(float(np.mean(sizes)), 2)
        else:
            avg_size = 0.0

        # ── Step 7: Assess gel effect on pores ──────────────────
        # When gel is applied, pores become less visible
        # because gel fills and covers them
        # High pore density = less gel coverage
        # Low pore density = more gel coverage (pores hidden)
        if pore_density < 0.05:
            gel_coverage_effect = "Pores mostly covered — good gel coverage"
        elif pore_density < 0.15:
            gel_coverage_effect = "Pores partially covered — moderate gel coverage"
        else:
            gel_coverage_effect = "Pores clearly visible — minimal gel coverage"

        # ── Step 8: Build summary ────────────────────────────────
        summary = (
            f"Detected {pore_count} pore-like features. "
            f"Density: {pore_density} per pixel. "
            f"{gel_coverage_effect}."
        )

        return {
            "pores_detected": pore_count > 0,
            "pore_count": pore_count,
            "pore_density": pore_density,
            "pore_size_avg": avg_size,
            "gel_coverage_effect": gel_coverage_effect,
            "summary": summary,
            "keypoints": keypoints,
            "hand_region": hand_region
        }

    def draw_pores(self, output_image, image_path, hand_boxes, pore_result):
        # Draw detected pores as red circles on the hand region
        if not pore_result["pores_detected"]:
            return output_image

        if not hand_boxes:
            return output_image

        best_hand = max(hand_boxes, key=lambda x: x["confidence"])
        x1, y1, x2, y2 = [int(c) for c in best_hand["coords"]]

        # Draw each pore as a small red circle
        for kp in pore_result["keypoints"]:
            # Adjust coordinates to full image position
            px = int(kp.pt[0]) + x1
            py = int(kp.pt[1]) + y1
            radius = max(int(kp.size / 2), 1)
            cv2.circle(output_image, (px, py), radius, (0, 0, 255), 1)

        return output_image 