# Task_6-KNN-
#  K-Nearest Neighbors (KNN) Classification – Breast Cancer Dataset

##  Objective

Implement the **K-Nearest Neighbors (KNN)** algorithm for binary classification using the Breast Cancer Wisconsin dataset. This task helps understand how KNN works, how feature scaling affects performance, and how to visualize decision boundaries.

---

## Dataset Used

**File:** `breast_cancer.csv`

**Target Column:** `diagnosis`
- `M` = Malignant (encoded as 1)
- `B` = Benign (encoded as 0)

**Features:** 30 numeric columns such as:
- `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean`, etc.

---

##  Tools & Libraries

- Python
- pandas
- scikit-learn
- matplotlib
- numpy

---

## Steps Followed

### 1. Load and Prepare Data
- Dropped the `id` column.
- Encoded `diagnosis` to binary (M → 1, B → 0).

### 2. Feature Scaling
- Normalized all features using `StandardScaler` to ensure KNN performs optimally.

### 3. Train/Test Split
- Dataset was split into 80% training and 20% testing sets.

### 4. Model Training and Evaluation
Used `KNeighborsClassifier` from `sklearn`. Models were trained with values of **K from 1 to 10**, and the following accuracies were recorded:

| K Value | Accuracy     |
|---------|--------------|
| 1       | 0.9386       |
| 2       | 0.9474       |
| 3       | 0.9474       |
| 4       | 0.9561       |
| 5       | 0.9474       |
| 6       | 0.9561       |
| 7       | 0.9474       |
| 8       | 0.9561       |
| 9       | **0.9649  Best** |
| 10      | 0.9561       |

### Confusion Matrix for Best K (K=9):

[[68 3]

[ 3 40]]

- **True Negatives:** 68 (Benign correctly predicted)
- **False Positives:** 3 (Benign misclassified as Malignant)
- **False Negatives:** 3 (Malignant misclassified as Benign)
- **True Positives:** 40 (Malignant correctly predicted)

---

##  What is KNN?

K-Nearest Neighbors (KNN) is a **non-parametric, lazy learning** algorithm:
- Classifies points based on the majority class among its `K` closest neighbors.
- Sensitive to feature scale, hence normalization is important.
- No training phase – the model simply stores the training data.

---

##  Visualization

### Decision Boundary Plot (K=5)

The plot shows the classification decision regions using just two features:
- `radius_mean` (X-axis)
- `texture_mean` (Y-axis)

![KNN Decision Boundary (K=5)]

- Red region: Predicted Malignant
- Blue region: Predicted Benign
- Dots: Actual data points (red = malignant, blue = benign)

---

##  Conclusion

- Best accuracy (96.49%) was achieved at **K = 9**.
- Normalization was crucial for optimal model performance.
- Visualizing the decision boundary helped understand class separability in feature space.
