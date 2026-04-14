# analyser.py
# This file calculates how clean the hand is based on gel coverage

class CleanlinessAnalyser:
    def __init__(self):
        pass

    def calculate_box_area(self, coords):
        # Calculate the area of a bounding box
        # coords = [x1, y1, x2, y2]
        x1, y1, x2, y2 = coords
        width = x2 - x1
        height = y2 - y1
        return width * height

    def calculate_overlap(self, hand_coords, gel_coords):
        # Calculate how much of the gel box overlaps with the hand box
        # This tells us how much gel is actually on the hand

        hx1, hy1, hx2, hy2 = hand_coords
        gx1, gy1, gx2, gy2 = gel_coords

        # Find the overlapping rectangle
        overlap_x1 = max(hx1, gx1)
        overlap_y1 = max(hy1, gy1)
        overlap_x2 = min(hx2, gx2)
        overlap_y2 = min(hy2, gy2)

        # Check if there is actually an overlap
        if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
            return 0.0  # no overlap at all

        # Calculate overlap area
        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
        return overlap_area

    def analyse(self, hand_boxes, gel_boxes):
        # If no hand detected at all
        if not hand_boxes:
            return {
                "score": 0.0,
                "result": "No hand detected",
                "hand_area": 0,
                "gel_area_on_hand": 0,
                "coverage_percent": 0.0
            }

        # Use the most confident hand detection
        best_hand = max(hand_boxes, key=lambda x: x["confidence"])
        hand_area = self.calculate_box_area(best_hand["coords"])

        # If no gel detected
        if not gel_boxes:
            return {
                "score": 0.0,
                "result": "0% clean — no gel detected",
                "hand_area": hand_area,
                "gel_area_on_hand": 0,
                "coverage_percent": 0.0
            }

        # Calculate total gel area that overlaps with the hand
        total_gel_on_hand = 0
        for gel in gel_boxes:
            overlap = self.calculate_overlap(
                best_hand["coords"],
                gel["coords"]
            )
            total_gel_on_hand += overlap

        # Calculate what percentage of the hand is covered by gel
        coverage_percent = (total_gel_on_hand / hand_area) * 100
        coverage_percent = min(coverage_percent, 100.0)  # cap at 100%

        # Convert gel coverage to cleanliness score
        # More gel coverage = cleaner hand
        cleanliness_score = round(coverage_percent, 1)

        # Give a label based on the score
        if cleanliness_score >= 70:
            label = "Clean"
        elif cleanliness_score >= 40:
            label = "Partially Clean"
        else:
            label = "Unclean"

        return {
            "score": cleanliness_score,
            "result": f"{cleanliness_score}% clean — {label}",
            "hand_area": hand_area,
            "gel_area_on_hand": total_gel_on_hand,
            "coverage_percent": coverage_percent
        } 