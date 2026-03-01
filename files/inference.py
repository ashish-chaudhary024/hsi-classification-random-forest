import joblib
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from datetime import datetime

# CONFIGURATION
MODEL_PATH = 'random_forest_model.pkl'
IMAGE_PATH = './HSI_img.tif' 

def run_classification():
    # 1. Load the pre-trained model
    print("--- Loading Pre-trained Random Forest Model ---")
    rf = joblib.load(MODEL_PATH)

    # 2. Load the HSI image
    with rasterio.open(IMAGE_PATH) as src:
        hsi_data = src.read()
        rows, cols = hsi_data.shape[1], hsi_data.shape[2]
        
        # Prepare RGB bands for visualization (e.g., bands 115, 72, 25)
        rgb = np.stack([src.read(115), src.read(72), src.read(25)], axis=-1)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

    # 3. Reshape for ML (Bands must be the last dimension)
    # New shape: (Pixels, Bands)
    hsi_reshaped = hsi_data.reshape(hsi_data.shape[0], -1).T

    # 4. Predict
    print(f"Classifying {rows * cols} pixels...")
    start = datetime.now()
    predictions = rf.predict(hsi_reshaped)
    print(f"Time taken: {datetime.now() - start}")

    # 5. Reshape back to 2D image
    classified_img = predictions.reshape(rows, cols)

    # 6. Visualize
    class_names = ['Sorghum', 'Soil', 'Concrete', 'Metal', 'Tarp']
    cmap = colors.ListedColormap(['green', 'orange', 'white', 'gray', 'blue'])
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb)
    plt.title("Original HSI (True Color)")
    
    plt.subplot(1, 2, 2)
    im = plt.imshow(classified_img, cmap=cmap)
    plt.title("Classification Result")
    plt.colorbar(im, ticks=[1, 2, 3, 4, 5]).ax.set_yticklabels(class_names)
    
    plt.show()

if __name__ == "__main__":
    run_classification()