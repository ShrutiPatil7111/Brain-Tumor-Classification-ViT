import tkinter as tk
from tkinter import filedialog, ttk

import torch
import torchvision.transforms as transforms
from transformers import ViTForImageClassification
import os
import torch.nn.functional as F
import numpy as np
from skimage import util
import sys

from PIL import Image, ImageTk

# Add this to help PyInstaller find resources
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Define the classes
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Define noise types
noise_types = ['No Noise', 'Gaussian', 'Salt & Pepper', 'Speckle']

# Define noise levels
noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]  # Levels for Gaussian and Speckle (scaled down)
noise_levels_for_sp = [0.05, 0.1, 0.15, 0.2, 0.25]  # Levels for Salt & Pepper

# Load the model
def load_model():
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=4, ignore_mismatched_sizes=True)
    model_path = resource_path("final_model.pth")
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)

# Noise functions
def add_gaussian_noise(image, noise_level):
    var = noise_level ** 2 * 0.01  # Scale down the variance for better control
    return util.random_noise(image, mode='gaussian', var=var)

def add_salt_and_pepper_noise(image, noise_level):
    return util.random_noise(image, mode='s&p', amount=noise_level)

def add_speckle_noise(image, noise_level):
    var = noise_level ** 2 * 0.01  # Scale down the variance for better control
    return util.random_noise(image, mode='speckle', var=var)

# Add noise to the image using defined functions
def add_noise(image, noise_type, noise_level):
    image_array = np.array(image) / 255.0  # Normalize image to [0, 1]

    if noise_type == 'Gaussian':
        noisy_image = add_gaussian_noise(image_array, noise_level)  # Use noise level directly
    elif noise_type == 'Salt & Pepper':
        amount = noise_levels_for_sp[noise_level - 1]  # Select based on the level
        noisy_image = add_salt_and_pepper_noise(image_array, amount)
    elif noise_type == 'Speckle':
        noisy_image = add_speckle_noise(image_array, noise_level)  # Use noise level directly
    else:
        return image  # No noise applied

    noisy_image = np.clip(noisy_image, 0, 1)  # Clip to [0, 1]
    return Image.fromarray((noisy_image * 255).astype(np.uint8))  # Convert back to PIL image

# Predict the class and return probabilities
def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor).logits
        probabilities = F.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()], probabilities

# Update the result label with the prediction
def update_result_label(prediction):
    color_map = {
        'glioma': 'red',
        'meningioma': 'blue',
        'notumor': 'green',
        'pituitary': 'purple',
    }
    result_label.config(text=f"Prediction: {prediction}", bg=color_map.get(prediction, 'blue'))

# Display probabilities in the table
def update_probabilities_table(probabilities):
    for item in table.get_children():
        table.delete(item)
    for i, cls in enumerate(classes):
        prob = probabilities[i].item()
        table.insert("", "end", values=(cls, f"{prob:.4f}"))

# Function to open an image and display it
def open_image():
    global current_image
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        current_image = image  # Store the original image for further processing
        display_image(current_image)

# Display the image on the UI
def display_image(image):
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo

# Predict image based on selected crop and noise options
def predict_and_display():
    if not current_image:
        return  # Ensure an image is loaded before predicting

    crop_type = crop_var.get()
    noise_type = noise_var.get()
    noise_level = noise_level_var.get()

    # Apply cropping based on selection
    width, height = current_image.size
    if crop_type == 'Left Half':
        cropped_image = current_image.crop((0, 0, width // 2, height))
    elif crop_type == 'Right Half':
        cropped_image = current_image.crop((width // 2, 0, width, height))
    elif crop_type == 'Top Half':
        cropped_image = current_image.crop((0, 0, width, height // 2))
    elif crop_type == 'Bottom Half':
        cropped_image = current_image.crop((0, height // 2, width, height))
    elif crop_type == 'Top-Right Quarter':
        cropped_image = current_image.crop((width // 2, 0, width, height // 2))
    elif crop_type == 'Top-Left Quarter':
        cropped_image = current_image.crop((0, 0, width // 2, height // 2))
    elif crop_type == 'Bottom-Right Quarter':
        cropped_image = current_image.crop((width // 2, height // 2, width, height))
    elif crop_type == 'Bottom-Left Quarter':
        cropped_image = current_image.crop((0, height // 2, width // 2, height))
    else:
        cropped_image = current_image  # No cropping

    # Ensure the cropped image is converted to RGB
    cropped_image = cropped_image.convert("RGB")

    # Add noise to the cropped image
    noisy_image = add_noise(cropped_image, noise_type, noise_level)

    # Display the noisy image
    display_image(noisy_image)

    # Preprocess the noisy image for prediction
    image_tensor = preprocess_image(noisy_image)
    prediction, probabilities = predict(model, image_tensor)
    update_result_label(prediction)
    update_probabilities_table(probabilities)

# Enable or disable noise level dropdown
def toggle_noise_level(*args):
    if noise_var.get() == "No Noise":
        noise_level_menu.config(state="disabled")
    else:
        noise_level_menu.config(state="normal")



# Create the main window
def main():
    global root, result_label, image_label, crop_var, noise_var, noise_level_var, noise_level_menu, table, current_image, model

    root = tk.Tk()
    root.title("Brain Tumor Detection")
    root.configure(bg="white") # Set background color
    
    '''
    # Call background setter
    set_background(root, "bg2.jpg")
    '''
    # Load the model
    model = load_model()

    # Create and pack the widgets
    frame = tk.Frame(root, bg="white")
    frame.pack(padx=10, pady=10)

    # Prediction label
    result_label = tk.Label(frame, text="Prediction", font=("Arial", 14), relief="solid", width=20, bg="blue", fg="white")
    result_label.pack(pady=5)

    # Image display area
    image_label = tk.Label(frame, bg="white")
    image_label.pack()

    # Buttons for image operations
    button_frame = tk.Frame(frame, bg="white")
    button_frame.pack(pady=5)

    open_button = tk.Button(button_frame, text="Open Image", command=open_image, width=15, bg="blue", fg="white")
    open_button.pack(side="left", padx=5)

    # Crop selection dropdown
    crop_var = tk.StringVar(value="No Cropping")
    crop_options = ["No Cropping", "Left Half", "Right Half", "Top Half", "Bottom Half", 
                    "Top-Right Quarter", "Top-Left Quarter", "Bottom-Right Quarter", "Bottom-Left Quarter"]
    crop_menu = tk.OptionMenu(button_frame, crop_var, *crop_options)
    crop_menu.config(width=15, bg="blue", fg="white")
    crop_menu.pack(side="left", padx=5)

    # Noise type selection dropdown
    noise_var = tk.StringVar(value="No Noise")
    noise_var.trace_add("write", toggle_noise_level)  # Add trace to toggle noise level dropdown
    noise_menu = tk.OptionMenu(button_frame, noise_var, *noise_types)
    noise_menu.config(width=15, bg="blue", fg="white")
    noise_menu.pack(side="left", padx=5)

    # Noise level selection dropdown
    noise_level_var = tk.IntVar(value=1)  # Default value
    noise_level_menu = tk.OptionMenu(button_frame, noise_level_var, *range(1, 6))
    noise_level_menu.config(width=15, bg="blue", fg="white", state="disabled")  # Initially disabled
    noise_level_menu.pack(side="left", padx=5)

    # Predict button
    predict_button = tk.Button(button_frame, text="Predict", command=predict_and_display, width=15, bg="blue", fg="white")
    predict_button.pack(side="left", padx=5)

    # Table to display probabilities
    table_frame = tk.Frame(root, bg="white")
    table_frame.pack(pady=10)

    table = ttk.Treeview(table_frame, columns=("Class", "Probability"), show='headings', height=4)
    table.heading("Class", text="Class")
    table.heading("Probability", text="Probability")
    table.pack()

    # Initialize current_image
    current_image = None

    # Start the GUI loop
    root.mainloop()

# Main entry point
if __name__ == "__main__":
    main()