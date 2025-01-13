# Brain-Tumor-Classification-ViT

This repository contains the code for a brain tumor classification system using Vision Transformers (ViTs), which is my capston project. The system is capable of classifying brain tumors into four categories: Meningioma, Glioma, Pituitary, and No Tumor.

**Key Features:**

* **Accurate Classification:** Achieves high accuracy in classifying brain tumors using a pre-trained ViT model.
* **Robustness to Noise:** Maintains high accuracy even with noisy or partially cropped MRI images.
* **User-Friendly Interface:** Provides an intuitive graphical user interface for easy interaction.
* **Cross-Platform Compatibility:** The application is packaged as an executable for easy deployment.

**Installation:**

1. **Download the executable file** from the releases section.
2. **Run the executable** on your Windows machine.

**Usage:**

1. **Load an MRI image** using the file browser.
2. **Select optional preprocessing steps:**
   - Cropping 
   - Noise addition (Gaussian, Salt & Pepper, Speckle)
3. **Click "Predict"** to classify the tumor.
4. **View the classification results:** The predicted tumor type and probabilities will be displayed.

**Dependencies:**

* Python
* PyTorch
* Torchvision
* Transformers
* Scikit-image
* Pillow
* Tkinter
* PyInstaller (for packaging)

**Note:**

* This project requires a compatible GPU for optimal performance. 
* The provided executable is specifically designed for Windows systems.

**Contributing:**

Contributions to this project are welcome. Please feel free to fork the repository and submit pull requests.


**Team:**

* Shruti Patil 
* Om Jadhav
* Khushi Deshmukh
* Atharv Kabade

Download from [here](https://github.com/ShrutiPatil7111/Brain-Tumor-Classification-ViT/releases/tag/v01)
