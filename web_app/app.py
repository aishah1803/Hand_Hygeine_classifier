from flask import Flask, request, jsonify, render_template
import random

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]

    return jsonify({
        "cleanliness": random.randint(0, 100)
    })

if __name__ == "__main__":
    app.run(debug=True)