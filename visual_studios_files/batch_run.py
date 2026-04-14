# batch_run.py
# Runs the pipeline on all test images and shows results for each one

import os
import cv2
from detector import HandGelDetector
from analyser import CleanlinessAnalyser
from colour_analyser import ColourAnalyser
from texture_analyser import TextureAnalyser
from scorer import CleanlinessScorer
from landmark_analyser import LandmarkAnalyser
from pore_analyser import PoreAnalyser

# ============================================================
# SETTINGS
# ============================================================
MODEL_PATH = "models/hand_det_yolo11.pt"
TEST_FOLDER = "test_images"
OUTPUT_FOLDER = "output/batch"

# ============================================================
# RUN ON ALL IMAGES
# ============================================================

def run_on_image(image_path, detector, analyser, colour_analyser,
                 texture_analyser, landmark_analyser, pore_analyser, scorer):

    # Detect
    hand_boxes, gel_boxes, result = detector.detect(image_path)

    # Analyse
    analysis = analyser.analyse(hand_boxes, gel_boxes)
    colour = colour_analyser.analyse_colour(image_path, hand_boxes, gel_boxes)
    texture = texture_analyser.analyse_texture(image_path, hand_boxes, gel_boxes)
    landmarks = landmark_analyser.analyse_landmarks(image_path, hand_boxes)
    pores = pore_analyser.analyse_pores(image_path, hand_boxes)

    # Score
    final = scorer.calculate_final_score(
        analysis, colour, texture, gel_boxes, pores
    )

    return {
        "image": os.path.basename(image_path),
        "hands": len(hand_boxes),
        "gel_regions": len(gel_boxes),
        "final_score": final["final_score"],
        "label": final["label"],
        "result": final["result"],
        "coverage": analysis["coverage_percent"],
        "pore_count": pores["pore_count"],
        "pore_density": pores["pore_density"],
        "brightness": colour["hand_colour"]["brightness"] if colour["hand_colour"] else 0,
        "consistency": texture["consistency"],
        "smoothness": texture["smoothness"],
        "fingers": landmarks["fingers_detected"] if landmarks["landmarks_found"] else [],
        "hand_position": landmarks["hand_position"] if landmarks["landmarks_found"] else "unknown",
        "breakdown": final["breakdown"]
    }

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load all modules once
    print("Loading model and modules...")
    detector = HandGelDetector(MODEL_PATH)
    analyser = CleanlinessAnalyser()
    colour_analyser = ColourAnalyser()
    texture_analyser = TextureAnalyser()
    landmark_analyser = LandmarkAnalyser()
    pore_analyser = PoreAnalyser()
    scorer = CleanlinessScorer()

    # Find all images in test folder
    valid_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [
        f for f in os.listdir(TEST_FOLDER)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]
    image_files.sort()

    print(f"Found {len(image_files)} images to process\n")

    all_results = []

    for filename in image_files:
        image_path = os.path.join(TEST_FOLDER, filename)
        print(f"Processing {filename}...")

        try:
            result = run_on_image(
                image_path, detector, analyser, colour_analyser,
                texture_analyser, landmark_analyser, pore_analyser, scorer
            )
            all_results.append(result)

            # Save output image with boxes
            output_image = detector.draw_boxes(
                image_path,
                [],  # rerun detect to get boxes
                []
            )
            hand_boxes, gel_boxes, _ = detector.detect(image_path)
            output_image = detector.draw_boxes(image_path, hand_boxes, gel_boxes)

            # Add result text on image
            cv2.putText(
                output_image,
                result["result"],
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            # Save
            out_path = os.path.join(OUTPUT_FOLDER, f"result_{filename}")
            cv2.imwrite(out_path, output_image)

        except Exception as e:
            print(f"  Error on {filename}: {e}")
            continue

    # Print summary table
    print("\n" + "="*70)
    print("BATCH RESULTS SUMMARY")
    print("="*70)
    print(f"{'Image':<15} {'Hands':<7} {'Gel':<5} {'Score':<8} {'Label':<20} {'Pores':<8}")
    print("-"*70)
    for r in all_results:
        print(f"{r['image']:<15} {r['hands']:<7} {r['gel_regions']:<5} "
              f"{r['final_score']:<8} {r['label']:<20} {r['pore_count']:<8}")
    print("="*70)

    return all_results

if __name__ == "__main__":
    main() 