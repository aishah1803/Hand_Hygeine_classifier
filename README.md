 Hand Hygiene Checker Website
   Overview

This project is a web-based prototype that simulates an AI system for evaluating hand hygiene using UV images. It estimates how clean hands are after washing by detecting remaining gel residue and displaying a percentage.

 Note: This website was not used in the final system due to integration issues, but it demonstrates the intended user interaction.

 Features
Upload hand image (UV light)
Analyze button with loading animation
Cleanliness percentage (0–100%)
Status: Clean / Moderate / Not Clean
Dynamic recommendations
Dark mode toggle
Example gallery
 System Flow
Frontend → Flask Backend → (Simulated AI) → Results Display
 Technologies
HTML, CSS, JavaScript
Python (Flask)
 How It Works
Upload image
Click Analyze
Image sent to /analyze
Backend returns a simulated percentage
UI updates with results
 Backend
Endpoint: /analyze (POST)
Input: image
Output:
{ "cleanliness": 75 }

Currently uses a random value (0–100), designed to be replaced with a real model.

 Run Locally
pip install flask
python app.py

Open:

http://127.0.0.1:5000
 Structure
index.html
style.css
script.js
app.py
README.md
 Limitations
No real AI model yet
Requires local server
Not deployed
 Future Work
Integrate AI model
Deploy online
Improve accuracy and UI
 Contribution
Frontend & Integration: Website design, UI, API connection
Model: Handled separately by teammate
 Summary

A functional prototype demonstrating the full pipeline from image upload to result display, ready for AI integration.
