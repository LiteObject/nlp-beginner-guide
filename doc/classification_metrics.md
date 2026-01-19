# Key Classification Metrics for Model Evaluation

In the context of supervised learning, evaluating a classification model typically involves the following key metrics:

## 1. Accuracy
The ratio of correctly predicted observations to the total observations.
*   **Formula:** `(True Positives + True Negatives) / Total Predictions`
*   **Use when:** Classes are balanced (e.g., 50% spam, 50% not spam).

## 2. Precision
The ratio of correctly predicted positive observations to the total predicted positives. It answers: "Of all instances predicted as positive, how many were actually positive?"
*   **Formula:** `True Positives / (True Positives + False Positives)`
*   **Use when:** The cost of False Positives is high (e.g., spam detection—you don't want to classify an important email as spam).

## 3. Recall (Sensitivity)
The ratio of correctly predicted positive observations to the all observations in the actual class. It answers: "Of all actual positive instances, how many did we predict correctly?"
*   **Formula:** `True Positives / (True Positives + False Negatives)`
*   **Use when:** The cost of False Negatives is high (e.g., cancer detection—you don't want to miss a patient who has cancer).

## 4. F1 Score
The *harmonic mean* of Precision and Recall. It balances the trade-off between the two. Unlike a simple average, the harmonic mean penalizes extreme differences between Precision and Recall, so both must be reasonably high for a good F1 score.
*   **Formula:** `2 * (Recall * Precision) / (Recall + Precision)`
*   **Use when:** You have an uneven class distribution or need a balance between Precision and Recall.

## 5. Confusion Matrix
A table that is often used to describe the performance of a classification model. It shows the actual values versus the predicted values, breaking down True Positives, True Negatives, False Positives, and False Negatives.

|                       | **Predicted Positive** | **Predicted Negative** |
|-----------------------|------------------------|------------------------|
| **Actual Positive**   | ✅ True Positive (TP)  | ❌ False Negative (FN) |
| **Actual Negative**   | ❌ False Positive (FP) | ✅ True Negative (TN)  |

*   **True Positive (TP):** Correctly predicted positive
*   **True Negative (TN):** Correctly predicted negative
*   **False Positive (FP):** Incorrectly predicted positive (Type I Error)
*   **False Negative (FN):** Incorrectly predicted negative (Type II Error)

## 6. AUC-ROC (Area Under The Curve - Receiver Operating Characteristics)
*   **ROC Curve:** A probability curve plotting the True Positive Rate (Recall) against the False Positive Rate at various threshold settings.
*   **AUC:** Measures the entire two-dimensional area underneath the ROC curve. It represents the degree or measure of separability. It tells how much the model is capable of distinguishing between classes.
