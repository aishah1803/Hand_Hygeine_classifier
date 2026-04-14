# metrics.py
# Evaluates the pipeline using multiple machine learning classifiers
# Includes: Logistic Regression, Random Forest, SVM, KNN
# Metrics: Accuracy, Precision, Recall, F1, ROC AUC, Confusion Matrix

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import LeaveOneOut

# ============================================================
# YOUR DATASET
# Each row = one image
# Features = what the pipeline measured
# Label = ground truth you confirmed
# ============================================================

# Features for each image in this order:
# [final_score, gel_regions, pore_count, pore_density,
#  brightness, consistency, smoothness, coverage]

data = {
    "images": [
        "test1.png",
        "test2.png",
        "test3.png",
        "test4.jpg",
        "test5.jpg",
        "test6.jpg"
    ],

    "features": np.array([
        # score  gel  pores   density  bright  consist smooth  coverage
        [36.7,   1,   11101,  1.8855,  223.5,  70.0,   94.0,   0.0   ],  # test1
        [53.8,   6,   354,    0.0597,  240.9,  78.6,   99.6,   1.8   ],  # test2
        [56.1,   6,   402,    0.0650,  235.0,  75.0,   98.0,   2.1   ],  # test3
        [24.3,   0,   2646,   0.3920,  180.0,  65.0,   90.0,   0.0   ],  # test4
        [18.5,   0,   7385,   1.8855,  223.5,  70.0,   94.0,   0.0   ],  # test5
        [61.0,   5,   37,     0.0120,  245.0,  80.0,   99.0,   3.5   ],  # test6
    ]),

    # Ground truth labels you confirmed
    "labels": [
        "Unclean",           # test1
        "Partially Clean",   # test2
        "Partially Clean",   # test3
        "Unclean",           # test4
        "Unclean",           # test5
        "Partially Clean",   # test6
    ]
}

# ============================================================
# SETUP
# ============================================================

os.makedirs("output/metrics", exist_ok=True)

# Encode labels to numbers
# Unclean = 0, Partially Clean = 1
le = LabelEncoder()
y = le.fit_transform(data["labels"])
X = data["features"]

# Scale features so all numbers are on same scale
# Important for Logistic Regression, SVM, KNN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Classes:", le.classes_)
print("Encoded labels:", y)
print(f"Dataset: {len(y)} images, {X.shape[1]} features each\n")

# ============================================================
# CLASSIFIERS
# ============================================================

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM":                  SVC(kernel='rbf', probability=True, random_state=42),
    "KNN":                  KNeighborsClassifier(n_neighbors=3)
}

# ============================================================
# EVALUATION
# We use Leave One Out cross validation
# Because we only have 6 images, this is the fairest approach
# It trains on 5 images and tests on 1, repeating for each image
# ============================================================

loo = LeaveOneOut()

print("="*60)
print("CLASSIFIER RESULTS")
print("="*60)

all_results = {}

for name, clf in classifiers.items():
    all_preds = []
    all_true = []
    all_proba = []

    for train_idx, test_idx in loo.split(X_scaled):
        X_train = X_scaled[train_idx]
        X_test = X_scaled[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Train and predict
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        proba = clf.predict_proba(X_test)[:, 1]

        all_preds.append(pred[0])
        all_true.append(y_test[0])
        all_proba.append(proba[0])

    # Calculate all metrics
    accuracy  = accuracy_score(all_true, all_preds)
    precision = precision_score(all_true, all_preds, average='weighted', zero_division=0)
    recall    = recall_score(all_true, all_preds, average='weighted', zero_division=0)
    f1        = f1_score(all_true, all_preds, average='weighted', zero_division=0)

    try:
        roc_auc = roc_auc_score(all_true, all_proba)
    except:
        roc_auc = 0.0

    cm = confusion_matrix(all_true, all_preds)

    all_results[name] = {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "roc_auc":   roc_auc,
        "cm":        cm,
        "preds":     all_preds,
        "true":      all_true
    }

    # Print results
    print(f"\n{name}")
    print("-"*40)
    print(f"  Accuracy:  {accuracy:.2%}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall:    {recall:.2%}")
    print(f"  F1 Score:  {f1:.2%}")
    print(f"  ROC AUC:   {roc_auc:.2f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        all_true, all_preds,
        target_names=le.classes_,
        zero_division=0
    ))

print("="*60)

# ============================================================
# PLOT 1 — Metrics comparison bar chart
# ============================================================

metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
clf_names = list(all_results.keys())

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(metrics_names))
width = 0.2

for i, (clf_name, results) in enumerate(all_results.items()):
    values = [
        results["accuracy"],
        results["precision"],
        results["recall"],
        results["f1"],
        results["roc_auc"]
    ]
    bars = ax.bar(x + i * width, values, width, label=clf_name)

ax.set_xlabel("Metric")
ax.set_ylabel("Score")
ax.set_title("Classifier Performance Comparison")
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metrics_names)
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("output/metrics/classifier_comparison.png", dpi=150)
print("\nSaved: output/metrics/classifier_comparison.png")

# ============================================================
# PLOT 2 — Confusion matrices for all classifiers
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (clf_name, results) in enumerate(all_results.items()):
    cm = results["cm"]
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=le.classes_
    )
    disp.plot(ax=axes[i], colorbar=False)
    axes[i].set_title(f"{clf_name}")

plt.suptitle("Confusion Matrices — All Classifiers", fontsize=14)
plt.tight_layout()
plt.savefig("output/metrics/confusion_matrices.png", dpi=150)
print("Saved: output/metrics/confusion_matrices.png")

# ============================================================
# PLOT 3 — Feature importance from Random Forest
# ============================================================

feature_names = [
    "Final Score",
    "Gel Regions",
    "Pore Count",
    "Pore Density",
    "Brightness",
    "Consistency",
    "Smoothness",
    "Coverage"
]

# Train Random Forest on full dataset for feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(len(feature_names)),
       importances[indices],
       color='steelblue')
ax.set_xticks(range(len(feature_names)))
ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
ax.set_title("Feature Importance — Random Forest")
ax.set_ylabel("Importance Score")
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("output/metrics/feature_importance.png", dpi=150)
print("Saved: output/metrics/feature_importance.png")

# ============================================================
# PLOT 4 — Per image prediction comparison
# ============================================================

fig, ax = plt.subplots(figsize=(12, 5))
image_names = data["images"]
true_labels = [le.classes_[t] for t in all_results["Random Forest"]["true"]]

x_pos = np.arange(len(image_names))
width = 0.2

for i, (clf_name, results) in enumerate(all_results.items()):
    pred_labels = [le.classes_[p] for p in results["preds"]]
    correct = [1 if p == t else 0
               for p, t in zip(pred_labels, true_labels)]
    ax.bar(x_pos + i * width, correct, width, label=clf_name)

ax.set_xticks(x_pos + width * 1.5)
ax.set_xticklabels(image_names, rotation=45, ha='right')
ax.set_yticks([0, 1])
ax.set_yticklabels(["Wrong", "Correct"])
ax.set_title("Per Image Prediction — Correct vs Wrong")
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("output/metrics/per_image_results.png", dpi=150)
print("Saved: output/metrics/per_image_results.png")

print("\nAll metrics and charts saved to output/metrics/")
print("Open that folder to see all 4 charts") 