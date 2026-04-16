let uploadedImage = null;
const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");

imageInput.addEventListener("change", function () {
    const file = this.files[0];

    if (file) {
        uploadedImage = file; // ✅ store image

        const reader = new FileReader();

        reader.onload = function (e) {
            preview.src = e.target.result;
        };

        reader.readAsDataURL(file);
    }
});

async function goToResults() {
    const resultsEl = document.getElementById("results");
    const loadingEl = document.getElementById("loading");
    const loadingTextEl = loadingEl.querySelector("p");
    const minimumVisibleMs = 900;

    const setLoadingState = (stateClass, text) => {
        loadingEl.classList.remove("loading--analyzing", "loading--warning", "loading--error");
        loadingEl.classList.add(stateClass);
        loadingTextEl.innerText = text;
        loadingEl.style.display = "block";
    };

    if (!uploadedImage) {
        setLoadingState("loading--warning", "Please upload an image first.");
        resultsEl.style.display = "none";
        setTimeout(() => {
            loadingEl.style.display = "none";
        }, 1400);
        return;
    }

    resultsEl.style.display = "none";
    setLoadingState("loading--analyzing", "Analyzing image...");
    const startedAt = Date.now();

    const formData = new FormData();
    formData.append("image", uploadedImage);

    try {
        const response = await fetch("http://127.0.0.1:5000/analyze", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Analyze request failed");
        }

        const data = await response.json();
        updateResults(data.cleanliness);

        const elapsed = Date.now() - startedAt;
        if (elapsed < minimumVisibleMs) {
            await new Promise((resolve) => setTimeout(resolve, minimumVisibleMs - elapsed));
        }

        loadingEl.style.display = "none";
        resultsEl.style.display = "block";
    } catch (error) {
        setLoadingState("loading--error", "Could not analyze image. Make sure Flask is running.");
        console.error(error);
        setTimeout(() => {
            loadingEl.style.display = "none";
        }, 2200);
    }
}

function updateResults(value) {
    let color;

    if (value >= 80) {
        color = "#4CAF50"; // green
    } else if (value >= 50) {
        color = "#ff9800"; // orange
    } else {
        color = "#f44336"; // red
    }

    document.querySelector(".circle").style.background =
        `conic-gradient(${color} 0% ${value}%, #e0e0e0 ${value}% 100%)`;

    // update text
    document.getElementById("percentage").innerText = value + "%";

    // update status
    let status = document.querySelector(".status");
    let recCards = document.querySelectorAll(".rec-card");

    if (value >= 80) {
        status.innerText = "Very Clean ✅";
        status.style.color = "#4CAF50";

        recCards[0].innerHTML = "<h3>👍 Great Job</h3><p>Your hands are very clean</p>";
        recCards[1].innerHTML = "<h3>✨ Maintain Hygiene</h3><p>Keep following proper handwashing</p>";
        recCards[2].innerHTML = "<h3>🧴 Optional</h3><p>You may use sanitizer for extra safety</p>";

    } else if (value >= 50) {
        status.innerText = "Partially Clean ⚠️";
        status.style.color = "#ff9800";

        recCards[0].innerHTML = "<h3>🧼 Wash Again</h3><p>Wash hands for at least 20 seconds</p>";
        recCards[1].innerHTML = "<h3>✋ Missed Areas</h3><p>Focus between fingers and nails</p>";
        recCards[2].innerHTML = "<h3>💧 Use Soap</h3><p>Ensure full soap coverage</p>";

    } else {
        status.innerText = "Not Clean ❌";
        status.style.color = "#f44336";

        recCards[0].innerHTML = "<h3>🚨 Rewash Immediately</h3><p>Your hands still have significant residue</p>";
        recCards[1].innerHTML = "<h3>🧼 Proper Technique</h3><p>Follow full handwashing steps</p>";
        recCards[2].innerHTML = "<h3>⏱️ Time</h3><p>Wash for at least 20 seconds thoroughly</p>";
    }
}


// DARK MODE TOGGLE
const toggle = document.getElementById("darkModeToggle");

toggle.addEventListener("change", () => {
    document.body.classList.toggle("dark");
});
