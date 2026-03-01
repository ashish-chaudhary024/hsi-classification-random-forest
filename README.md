# HSI Classification: Random Forest (99% Accuracy)

This repository contains a pre-trained **Random Forest** model for the pixel-wise classification of Hyperspectral Images (HSI). The model is optimized to distinguish between agricultural crops and various urban/industrial materials.



---

## 📊 Project Overview
In this project, I evaluated four machine learning models to classify HSI data into five distinct categories. The **Random Forest** model outperformed others (Logistic Regression, SVM, and Decision Trees) with a near-perfect accuracy.

### Target Classes:
1. **Sorghum** (Green)
2. **Soil** (Orange)
3. **Concrete** (White)
4. **Metal** (Gray)
5. **Tarp** (Blue)

---

## 🛠️ Repository Contents
* `random_forest_model.pkl`: The saved model file (Joblib format).
* `inference.py`: Python script to load the model and classify your own `.tif` images.
* `requirements.txt`: List of required Python packages.

---

## 🚀 How to Use the Model

Follow these steps to implement the model on your own hyperspectral data:

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/hsi-classification-random-forest.git](https://github.com/YOUR_USERNAME/hsi-classification-random-forest.git)
cd hsi-classification-random-forest
