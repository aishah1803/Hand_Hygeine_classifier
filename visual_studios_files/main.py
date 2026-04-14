# main.py
# This is the file you run to analyse a hand image
# It ties together all analysis modules into one final score

import cv2
import os
from detector import HandGelDetector
from analyser import CleanlinessAnalyser
from colour_analyser import ColourAnalyser
from texture_analyser import TextureAnalyser
from scorer import CleanlinessScorer
from landmark_analyser import LandmarkAnalyser
from pore_analyser import PoreAnalyser

# ============================================================
# SETTINGS - change these to match your files
# ============================================================

# Path to your trained YOLO11 model
MODEL_PATH = "models/hand_det_yolo11.pt"

# Path to the image you want to analyse
IMAGE_PATH = "test_images/test3.jpg"

# Where to save the output image
OUTPUT_PATH = "output/result3.jpg"

# ============================================================
# MAIN CODE - you do not need to change anything below here
# ============================================================

def main():
    # Step 1 - make sure output folders exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("test_images", exist_ok=True)

    # Step 2 - check the image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"Image not found at: {IMAGE_PATH}")
        print("Please put a test image into the test_images folder")
        return

    # Step 3 - load the detector
    print("Loading YOLO11 model...")
    detector = HandGelDetector(MODEL_PATH)

    # Step 4 - detect hands and gel
    print("Detecting hands and gel in image...")
    hand_boxes, gel_boxes, result = detector.detect(IMAGE_PATH)
    print(f"Found {len(hand_boxes)} hand(s) and {len(gel_boxes)} gel region(s)")

    # Step 5 - analyse cleanliness from box coverage
    print("Analysing cleanliness...")
    analyser = CleanlinessAnalyser()
    analysis = analyser.analyse(hand_boxes, gel_boxes)

    # Step 6 - analyse colour
    print("Analysing colour...")
    colour_analyser = ColourAnalyser()
    colour = colour_analyser.analyse_colour(IMAGE_PATH, hand_boxes, gel_boxes)

    # Step 7 - analyse texture
    print("Analysing texture...")
    texture_analyser = TextureAnalyser()
    texture = texture_analyser.analyse_texture(IMAGE_PATH, hand_boxes, gel_boxes)

    # Step 8 - analyse landmarks
    print("Analysing hand landmarks...")
    landmark_analyser = LandmarkAnalyser()
    landmarks = landmark_analyser.analyse_landmarks(IMAGE_PATH, hand_boxes)
    if landmarks['landmarks_found']:
        print(f"Fingers extended: {landmarks['fingers_detected']}")

    # Step 9 - analyse pores
    print("Analysing pores...")
    pore_analyser = PoreAnalyser()
    pores = pore_analyser.analyse_pores(IMAGE_PATH, hand_boxes)
    print(f"Pores detected: {pores['pore_count']}")

    # Step 10 - combine all features into final score
    print("Calculating final score...")
    scorer = CleanlinessScorer()
    final = scorer.calculate_final_score(
        analysis, colour, texture, gel_boxes, pores
    )

    # Step 11 - print full report
    print("\n" + "="*40)
    print("CLEANLINESS RESULT")
    print("="*40)
    print(f"  {final['result']}")
    print("="*40)
    print("\nSCORE BREAKDOWN")
    print("="*40)
    print(f"  Gel coverage score:   {final['breakdown']['coverage_score']}/100")
    print(f"  Colour diff score:    {final['breakdown']['brightness_score']}/100")
    print(f"  Consistency score:    {final['breakdown']['consistency_score']}/100")
    print(f"  Detection confidence: {final['breakdown']['confidence_score']}/100")
    print(f"  Pore score:           {final['breakdown']['pore_score']}/100")
    print("="*40)
    print("\nLANDMARK ANALYSIS")
    print("="*40)
    if landmarks['landmarks_found']:
        print(f"  Summary:     {landmarks['summary']}")
        print(f"  Fingers:     {', '.join(landmarks['fingers_detected']) if landmarks['fingers_detected'] else 'none extended'}")
        print(f"  Hand spread: {landmarks['hand_coverage']}%")
        print(f"  Position:    {landmarks['hand_position']}")
    else:
        print(f"  {landmarks['summary']}")
    print("="*40)
    print("\nPORE ANALYSIS")
    print("="*40)
    print(f"  Pore count:    {pores['pore_count']}")
    print(f"  Pore density:  {pores['pore_density']}")
    print(f"  Avg pore size: {pores['pore_size_avg']} px")
    print(f"  Gel effect:    {pores['gel_coverage_effect']}")
    print(f"  Summary:       {pores['summary']}")
    print("="*40)
    print("\nDETAILED ANALYSIS")
    print("="*40)
    print(f"  Gel coverage:    {analysis['coverage_percent']:.1f}%")
    print(f"  Colour summary:  {colour['summary']}")
    print(f"  Texture summary: {texture['summary']}")
    print("="*40)

    # Step 12 - draw landmarks on image first
    output_image = landmark_analyser.draw_landmarks(IMAGE_PATH, landmarks)

    # Draw YOLO boxes on top
    output_image = detector.draw_boxes(IMAGE_PATH, hand_boxes, gel_boxes)

    # Draw pores on top
    output_image = pore_analyser.draw_pores(
        output_image, IMAGE_PATH, hand_boxes, pores
    )

    # Add main result text
    cv2.putText(
        output_image,
        final['result'],
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # Add score breakdown
    cv2.putText(
        output_image,
        f"Coverage: {final['breakdown']['coverage_score']}%  |  Consistency: {final['breakdown']['consistency_score']}%",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2
    )

    # Add pore info
    cv2.putText(
        output_image,
        f"Pores: {pores['pore_count']} — {pores['gel_coverage_effect']}",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2
    )

    # Add landmark info
    if landmarks['landmarks_found']:
        cv2.putText(
            output_image,
            f"Hand: {landmarks['hand_position']} — {landmarks['num_fingers_extended']} fingers extended",
            (10, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # Add gel spread
    cv2.putText(
        output_image,
        colour['gel_spread'],
        (10, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2
    )

    # Add pore score
    cv2.putText(
        output_image,
        f"Pore score: {final['breakdown']['pore_score']}/100",
        (10, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 255),
        2
    )

    # Save output image
    cv2.imwrite(OUTPUT_PATH, output_image)
    print(f"\nOutput image saved to: {OUTPUT_PATH}")
    print("Open the output folder to see the result image")

if __name__ == "__main__":
    main() 