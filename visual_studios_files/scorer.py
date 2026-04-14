# scorer.py
# Combines all features into one final cleanliness score
# Uses a weighted scoring system including pore analysis

class CleanlinessScorer:
    def __init__(self):
        # These weights decide how much each feature matters
        # They all add up to 1.0 (100%)
        self.weights = {
            "coverage":     0.20,
            "brightness":   0.25,
            "consistency":  0.20,
            "confidence":   0.15,
            "pore":         0.20
        }

    def calculate_final_score(self, analysis, colour, texture, gel_boxes, pores=None):

        # ── Feature 1: Gel coverage score ───────────────────────
        # How much of the hand is covered by gel box
        # Bonus points for multiple gel detections
        coverage_score = min(analysis["coverage_percent"], 100.0)
        if gel_boxes:
            num_gel_regions = len(gel_boxes)
            gel_bonus = min((num_gel_regions - 1) * 15, 60)
            coverage_score = min(coverage_score + gel_bonus, 100.0)

        # ── Feature 2: Brightness difference score ──────────────
        # Big brightness difference = gel clearly visible on hand
        brightness_score = 0.0
        if colour["hand_colour"] and colour["gel_colour"]:
            hand_brightness = colour["hand_colour"]["brightness"]
            gel_brightness = colour["gel_colour"]["brightness"]
            brightness_diff = abs(hand_brightness - gel_brightness)
            brightness_score = min((brightness_diff / 80) * 100, 100.0)
        elif colour["hand_colour"] and not colour["gel_colour"]:
            brightness_score = 10.0

        # ── Feature 3: Texture consistency score ────────────────
        # Higher consistency = gel spread more evenly
        consistency_score = texture["consistency"]

        # ── Feature 4: YOLO gel confidence score ────────────────
        # How confident YOLO was when it detected gel
        confidence_score = 0.0
        if gel_boxes:
            best_gel = max(gel_boxes, key=lambda x: x["confidence"])
            confidence_score = best_gel["confidence"] * 100
        else:
            confidence_score = 0.0

        # ── Feature 5: Pore score ────────────────────────────────
        # Fewer pores visible = more gel coverage = cleaner hand
        # Gel fills and covers pores so lower density = better coverage
        pore_score = 50.0  # default if no pore data available
        if pores and pores["pores_detected"]:
            density = pores["pore_density"]
            if density < 0.05:
                pore_score = 90.0   # very few pores = very well covered
            elif density < 0.15:
                pore_score = 65.0   # some pores = moderate coverage
            elif density < 0.5:
                pore_score = 35.0   # many pores = low coverage
            else:
                pore_score = 10.0   # very many pores = no gel at all
        elif pores and not pores["pores_detected"]:
            pore_score = 85.0       # no pores detected = very well covered

        # ── Combine all scores using weights ────────────────────
        final_score = (
            coverage_score    * self.weights["coverage"] +
            brightness_score  * self.weights["brightness"] +
            consistency_score * self.weights["consistency"] +
            confidence_score  * self.weights["confidence"] +
            pore_score        * self.weights["pore"]
        )

        final_score = round(final_score, 1)

        # ── Give a label based on score ──────────────────────────
        if final_score >= 70:
            label = "Clean"
        elif final_score >= 40:
            label = "Partially Clean"
        else:
            label = "Unclean"

        return {
            "final_score": final_score,
            "label": label,
            "result": f"This hand is {final_score}% clean — {label}",
            "breakdown": {
                "coverage_score":     round(coverage_score, 1),
                "brightness_score":   round(brightness_score, 1),
                "consistency_score":  round(consistency_score, 1),
                "confidence_score":   round(confidence_score, 1),
                "pore_score":         round(pore_score, 1)
            }
        } 