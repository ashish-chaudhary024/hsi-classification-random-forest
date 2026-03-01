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
* `HSI_Classificaion_Result.png`: Reference classified map.



---

## 🚀 How to Use the Model

Follow these steps to implement the model on your own hyperspectral data:

### 1. Clone the Repository
```bash

git clone [https://github.com/ashish-chaudhary024/hsi-classification-random-forest.git]
cd hsi-classification-random-forest```
```
--- 
### 2. Repository Structure
Ensure your folder contains:  
* `README.md`: The project overview
* `LICENSE`: The MIT permissions
* `requirements.txt`: The library list
* `random_forest_model.pkl`: The trained model
* `inference.py`: The Python script provided
---

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
---
### 4. Run Inference
```bash
python inference.py
```
---
## 🎓 Acknowledgments  

This project was developed as part of the coursework under **Dr. Maitiniyazi Maimaitijiang** at South Dakota State University. Special thanks to the instructor for the guidance on hyperspectral data processing and machine learning applications in remote sensing.

---
## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

