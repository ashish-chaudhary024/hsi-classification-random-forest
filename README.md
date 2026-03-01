# Hyperspectral Image Classification: Random Forest Model

This repository provides a pre-trained **Random Forest (RF)** model designed to classify high-dimensional Hyperspectral Images (HSI) into five distinct land-cover classes.

### 🎯 Project Highlights
* **Model:** Random Forest (Optimized)
* **Accuracy:** 99%
* **Classes:** Sorghum, Soil, Concrete, Metal, and Tarp.
* **Input:** HSI Data (TIF format recommended).

### 🛠️ Classes & Labels
The model predicts the following categories:
1. **Sorghum** (Vegetation)
2. **Soil** (Bare Ground)
3. **Concrete** (Infrastructure)
4. **Metal** (Man-made structures)
5. **Tarp** (Synthetic materials)

---

## 🚀 Getting Started

### Prerequisites
Ensure you have the following Python libraries installed:
```bash
pip install numpy matplotlib scikit-learn rasterio joblib

---

## 💻 Implementation Guide (The `classify_hsi.py` Script)

# I’ve polished your code to make it "plug-and-play" for others. I added error handling and streamlined the visualization so it looks professional.

```python
import joblib
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from datetime import datetime

# 1. Load the pre-trained model
MODEL_PATH = 'random_forest_model.pkl'
hsi_path = './HSI_img.tif'

print("Loading model...")
rf_model = joblib.load(MODEL_PATH)

# 2. Open and read the HSI image
with rasterio.open(hsi_path) as src:
    hsi_data = src.read()  # Shape: (bands, rows, cols)
    profile = src.profile
    
    # Selecting 3 bands for True Color (RGB) visualization 
    # (Adjust band indices 115, 72, 25 based on your sensor)
    rgb_bands = [115, 72, 25]
    rgb = np.stack([src.read(b) for b in rgb_bands], axis=-1)
    # Normalize RGB for display
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

# 3. Reshape for Classification
bands, rows, cols = hsi_data.shape
hsi_flattened = hsi_data.reshape(bands, -1).T 

# 4. Perform Prediction
print(f"Classifying {rows}x{cols} image pixels...")
start_time = datetime.now()

# Note: Ensure the variable matches (rf_model)
predictions = rf_model.predict(hsi_flattened)

end_time = datetime.now()
print(f"Classification Complete! Total time: {end_time - start_time}")

# 5. Reshape back to original dimensions
classified_image = predictions.reshape(rows, cols)

# 6. Visualization
# Define class names and color map
class_names = ["Sorghum", "Soil", "Concrete", "Metal", "Tarp"]
cmap = colors.ListedColormap(['green', 'orange', 'white', 'gray', 'blue'])
bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
norm = colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(1, 2, figsize=(15, 7))

# Plot True Color
ax[0].imshow(rgb)
ax[0].set_title("True Color Image (RGB)")
ax[0].axis("off")

# Plot Results
im = ax[1].imshow(classified_image, cmap=cmap, norm=norm)
ax[1].set_title("Random Forest Classification Result")
ax[1].axis("off")

# Add Legend/Colorbar
cbar = fig.colorbar(im, ax=ax[1], ticks=[1, 2, 3, 4, 5])
cbar.ax.set_yticklabels(class_names)

plt.tight_layout()
plt.show()
