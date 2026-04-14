# gui.py
# Graphical User Interface for the Hand Hygiene Analyser

import tkinter as tk
from tkinter import filedialog
import cv2
import os
import threading
from PIL import Image, ImageTk

from detector import HandGelDetector
from analyser import CleanlinessAnalyser
from colour_analyser import ColourAnalyser
from texture_analyser import TextureAnalyser
from scorer import CleanlinessScorer
from landmark_analyser import LandmarkAnalyser
from pore_analyser import PoreAnalyser

MODEL_PATH = "models/hand_det_yolo11.pt"

class HandHygieneApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Hygiene Analyser")
        self.root.geometry("900x750")
        self.root.configure(bg="#1e1e2e")
        self.image_path = None
        self.output_path = "output/gui_result.jpg"
        os.makedirs("output", exist_ok=True)
        self.load_modules()
        self.build_ui()

    def load_modules(self):
        self.detector = HandGelDetector(MODEL_PATH)
        self.analyser = CleanlinessAnalyser()
        self.colour_analyser = ColourAnalyser()
        self.texture_analyser = TextureAnalyser()
        self.landmark_analyser = LandmarkAnalyser()
        self.pore_analyser = PoreAnalyser()
        self.scorer = CleanlinessScorer()

    def build_ui(self):
        # ── Title ────────────────────────────────────────────
        title_frame = tk.Frame(self.root, bg="#313244", pady=15)
        title_frame.pack(fill="x")

        tk.Label(
            title_frame,
            text="Hand Hygiene Analyser",
            font=("Helvetica", 22, "bold"),
            bg="#313244",
            fg="#cdd6f4"
        ).pack()

        tk.Label(
            title_frame,
            text="Upload a hand image to analyse gel coverage and cleanliness",
            font=("Helvetica", 11),
            bg="#313244",
            fg="#a6adc8"
        ).pack()

        # ── Controls ─────────────────────────────────────────
        controls_frame = tk.Frame(self.root, bg="#1e1e2e", pady=10)
        controls_frame.pack(fill="x", padx=20)

        self.file_label = tk.Label(
            controls_frame,
            text="No image selected",
            font=("Helvetica", 10),
            bg="#313244",
            fg="#a6adc8",
            width=50,
            anchor="w",
            padx=10
        )
        self.file_label.pack(side="left", padx=(0, 10), ipady=5)

        tk.Button(
            controls_frame,
            text="Browse Image",
            command=self.browse_image,
            bg="#89b4fa",
            fg="#1e1e2e",
            font=("Helvetica", 10, "bold"),
            padx=15, pady=5,
            relief="flat",
            cursor="hand2"
        ).pack(side="left", padx=(0, 10))

        self.analyse_btn = tk.Button(
            controls_frame,
            text="Analyse Hand",
            command=self.run_analysis,
            bg="#a6e3a1",
            fg="#1e1e2e",
            font=("Helvetica", 10, "bold"),
            padx=15, pady=5,
            relief="flat",
            cursor="hand2",
            state="disabled"
        )
        self.analyse_btn.pack(side="left")

        # ── Content area ─────────────────────────────────────
        content_frame = tk.Frame(self.root, bg="#1e1e2e")
        content_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Left — image
        left_frame = tk.Frame(content_frame, bg="#313244", width=450)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        left_frame.pack_propagate(False)

        tk.Label(
            left_frame,
            text="Result Image",
            font=("Helvetica", 11, "bold"),
            bg="#313244",
            fg="#cdd6f4"
        ).pack(pady=(10, 5))

        self.image_label = tk.Label(
            left_frame,
            bg="#313244",
            text="Image will appear here",
            fg="#6c7086",
            font=("Helvetica", 12)
        )
        self.image_label.pack(expand=True)

        # Right — results
        right_frame = tk.Frame(content_frame, bg="#1e1e2e", width=380)
        right_frame.pack(side="right", fill="both", padx=(10, 0))
        right_frame.pack_propagate(False)

        # Result box
        result_box = tk.Frame(right_frame, bg="#313244", pady=15)
        result_box.pack(fill="x", pady=(0, 10))

        tk.Label(
            result_box,
            text="CLEANLINESS RESULT",
            font=("Helvetica", 10, "bold"),
            bg="#313244",
            fg="#a6adc8"
        ).pack()

        self.result_label = tk.Label(
            result_box,
            text="—",
            font=("Helvetica", 24, "bold"),
            bg="#313244",
            fg="#cdd6f4"
        )
        self.result_label.pack(pady=5)

        self.label_badge = tk.Label(
            result_box,
            text="",
            font=("Helvetica", 13, "bold"),
            bg="#313244",
            fg="#a6e3a1"
        )
        self.label_badge.pack()

        # Score bars
        breakdown_box = tk.Frame(right_frame, bg="#313244", pady=10)
        breakdown_box.pack(fill="x", pady=(0, 10))

        tk.Label(
            breakdown_box,
            text="SCORE BREAKDOWN",
            font=("Helvetica", 10, "bold"),
            bg="#313244",
            fg="#a6adc8"
        ).pack(pady=(0, 5))

        self.score_bars = {}
        score_items = [
            ("Coverage",    "#89b4fa"),
            ("Colour",      "#f38ba8"),
            ("Consistency", "#a6e3a1"),
            ("Confidence",  "#fab387"),
            ("Pore Score",  "#cba6f7")
        ]

        for label, colour in score_items:
            row = tk.Frame(breakdown_box, bg="#313244")
            row.pack(fill="x", padx=15, pady=2)

            tk.Label(
                row,
                text=f"{label}:",
                font=("Helvetica", 9),
                bg="#313244",
                fg="#cdd6f4",
                width=12,
                anchor="w"
            ).pack(side="left")

            bar_bg = tk.Frame(row, bg="#45475a", height=12, width=150)
            bar_bg.pack(side="left", padx=5)
            bar_bg.pack_propagate(False)

            bar_fill = tk.Frame(bar_bg, bg=colour, height=12, width=0)
            bar_fill.place(x=0, y=0, height=12)

            score_lbl = tk.Label(
                row,
                text="0/100",
                font=("Helvetica", 9),
                bg="#313244",
                fg="#cdd6f4"
            )
            score_lbl.pack(side="left")

            self.score_bars[label] = (bar_fill, score_lbl)

        # Details box
        details_box = tk.Frame(right_frame, bg="#313244", pady=10)
        details_box.pack(fill="x", pady=(0, 10))

        tk.Label(
            details_box,
            text="DETAILS",
            font=("Helvetica", 10, "bold"),
            bg="#313244",
            fg="#a6adc8"
        ).pack(pady=(0, 5))

        self.details_text = tk.Text(
            details_box,
            height=9,
            bg="#313244",
            fg="#cdd6f4",
            font=("Courier", 9),
            relief="flat",
            state="disabled",
            padx=10,
            pady=5
        )
        self.details_text.pack(fill="x", padx=10)

        # Status bar
        self.status_label = tk.Label(
            self.root,
            text="Ready — select an image to begin",
            font=("Helvetica", 9),
            bg="#313244",
            fg="#a6adc8",
            anchor="w",
            padx=10
        )
        self.status_label.pack(fill="x", side="bottom", ipady=5)

    def browse_image(self):
        path = filedialog.askopenfilename(
            title="Select a hand image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.image_path = path
            self.file_label.config(text=os.path.basename(path))
            self.analyse_btn.config(state="normal")
            self.status_label.config(text=f"Image loaded: {os.path.basename(path)}")
            self.show_image(path)

    def show_image(self, path):
        img = Image.open(path)
        img.thumbnail((420, 400))
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo

    def update_score_bar(self, label, score):
        bar_fill, score_lbl = self.score_bars[label]
        width = int((score / 100) * 150)
        bar_fill.place(x=0, y=0, height=12, width=width)
        score_lbl.config(text=f"{score}/100")

    def run_analysis(self):
        self.analyse_btn.config(state="disabled", text="Analysing...")
        self.status_label.config(text="Analysing image — please wait...")
        thread = threading.Thread(target=self.analyse_image)
        thread.start()

    def analyse_image(self):
        try:
            hand_boxes, gel_boxes, _ = self.detector.detect(self.image_path)
            analysis  = self.analyser.analyse(hand_boxes, gel_boxes)
            colour    = self.colour_analyser.analyse_colour(self.image_path, hand_boxes, gel_boxes)
            texture   = self.texture_analyser.analyse_texture(self.image_path, hand_boxes, gel_boxes)
            landmarks = self.landmark_analyser.analyse_landmarks(self.image_path, hand_boxes)
            pores     = self.pore_analyser.analyse_pores(self.image_path, hand_boxes)
            final     = self.scorer.calculate_final_score(analysis, colour, texture, gel_boxes, pores)

            # Save output image
            output_image = self.detector.draw_boxes(self.image_path, hand_boxes, gel_boxes)
            output_image = self.pore_analyser.draw_pores(output_image, self.image_path, hand_boxes, pores)
            cv2.putText(output_image, final['result'],
                       (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 255), 2)
            cv2.imwrite(self.output_path, output_image)

            self.root.after(0, self.update_ui, final, analysis, colour, texture, landmarks, pores)

        except Exception as e:
            self.root.after(0, self.show_error, str(e))

    def update_ui(self, final, analysis, colour, texture, landmarks, pores):
        # Update result
        score = final["final_score"]
        label = final["label"]
        self.result_label.config(text=f"{score}% Clean")

        colour_code = {"Clean": "#a6e3a1", "Partially Clean": "#f9e2af", "Unclean": "#f38ba8"}.get(label, "#cdd6f4")
        self.label_badge.config(text=label, fg=colour_code)

        # Update score bars
        bd = final["breakdown"]
        self.update_score_bar("Coverage",    bd["coverage_score"])
        self.update_score_bar("Colour",      bd["brightness_score"])
        self.update_score_bar("Consistency", bd["consistency_score"])
        self.update_score_bar("Confidence",  bd["confidence_score"])
        self.update_score_bar("Pore Score",  bd["pore_score"])

        # Update details
        fingers = ', '.join(landmarks.get('fingers_detected', [])) or 'none'
        hand_pos = landmarks.get('hand_position', 'unknown')
        details = (
            f"Gel regions:      {len([g for g in [bd['confidence_score']] if g > 0])}\n"
            f"Gel coverage:     {analysis['coverage_percent']:.1f}%\n"
            f"Pore count:       {pores['pore_count']}\n"
            f"Pore density:     {pores['pore_density']}\n"
            f"Gel effect:       {pores['gel_coverage_effect']}\n"
            f"Hand position:    {hand_pos}\n"
            f"Fingers extended: {fingers}\n"
            f"Colour:           {colour['gel_spread']}\n"
            f"Texture:          {texture['summary'][:40]}\n"
        )

        self.details_text.config(state="normal")
        self.details_text.delete("1.0", "end")
        self.details_text.insert("end", details)
        self.details_text.config(state="disabled")

        # Show result image
        self.show_image(self.output_path)

        # Re-enable button
        self.analyse_btn.config(state="normal", text="Analyse Hand")
        self.status_label.config(text=f"Analysis complete — {final['result']}")

    def show_error(self, error_msg):
        self.status_label.config(text=f"Error: {error_msg}")
        self.analyse_btn.config(state="normal", text="Analyse Hand")
        print(f"Error: {error_msg}")


# ============================================================
# RUN THE APP
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = HandHygieneApp(root)
    root.mainloop() 