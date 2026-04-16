# Hand Hygiene Web App Prototype 🌐

This folder contains the web interface and backend server for our Hand Hygiene AI project. 
Users can upload an image of their hands under UV light to receive a cleanliness score and recommendations.

⚠️ **IMPORTANT NOTE: THE RESULTS ARE CURRENTLY FAKE** ⚠️
> The actual YOLOv26 AI model is **not** connected to this website yet. Right now, when you click "Analyze", the backend uses a Python script to generate a random, fake percentage just to test the UI. We did this to make sure the loading animations, dark mode, and result screens work perfectly before we plug in the heavy AI model. 

## 🛠️ Built With
* **Frontend:** HTML, CSS, JavaScript 
* **Backend:** Python, Flask
* **AI Model:** *(Pending - currently returning mocked random numbers)*

## 🚀 How to Run the Website
To run this website on your own computer, follow these steps:

1. Open your terminal and navigate to this `web_app` folder.
2. Install Flask by running: 
   `pip install Flask`
3. Start the server by running: 
   `python app.py`
4. Open your web browser and go to: `http://127.0.0.1:5000`

## ✨ Features
* **Image Upload:** Send UV hand images to the backend via a FormData POST request.
* **Dynamic UI:** Displays cleanliness percentage, status (Clean / Moderate / Not Clean), and smart recommendations.
* **Extras:** Dark mode toggle, loading animations, and an example gallery.
