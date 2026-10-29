
#  Plant Leaf Disease Detection Using Deep Learning

##  Project Overview

Plant Leaf Disease Detection is a deep learning project that identifies potential plant diseases from leaf images using a Convolutional Neural Network (CNN).

The project uses the PlantVillage dataset to train an image classification model with TensorFlow and Keras. A Streamlit web application allows users to upload a leaf image and view the predicted class and model confidence.

##  Objectives

- Identify plant leaf disease categories from images.
- Train a CNN model using labeled leaf images.
- Display predictions through a simple web interface.
- Explore how deep learning can support agricultural monitoring.

## Features

- Upload leaf images in JPG, JPEG, or PNG format.
- Classify images into trained plant disease categories.
- Display prediction confidence.
- Train and save a TensorFlow model.
- Save class labels for interpreting model predictions.

## Technologies Used

- **Programming Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Model:** Convolutional Neural Network (CNN)
- **Web Interface:** Streamlit
- **Image Processing:** Pillow
- **Numerical Computing:** NumPy
- **Dataset:** PlantVillage

##  Project Structure

```text
plant-leaf-disease-detection/
├── dataset/                 # Local dataset (not uploaded)
├── models/                  # Locally trained model (not uploaded)
│   ├── plant_disease_model.keras
│   └── class_names.json
├── app.py                   # Streamlit web application
├── train_model.py           # Model training script
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

## 📊 Dataset

This project uses the PlantVillage plant leaf image dataset.

Dataset link: https://www.kaggle.com/datasets/emmarex/plantdisease

Download and extract the dataset. Place the individual plant disease class folders inside the `dataset/` directory.

The training script automatically detects class folders and splits the images into training and validation sets.

The dataset is not included in this GitHub repository because of its size.

##  Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Yashaswinimadhava/plant-leaf-disease-detection.git
cd plant-leaf-disease-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the Dataset

Download the PlantVillage dataset and place the extracted disease-class folders inside `dataset/`.

### 5. Train the Model

```bash
python train_model.py
```

The training script saves the trained model and class labels inside the `models/` directory.

### 6. Run the Web Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal to use the application.

##  Model Details

The project uses a CNN with convolutional layers, pooling, data augmentation, and dense layers.

- **Input image size:** 224 × 224 pixels
- **Optimizer:** Adam
- **Loss function:** Sparse categorical cross-entropy
- **Evaluation metric:** Accuracy
- **Output layer:** Softmax classification across the dataset's classes

## Evaluation

The model reports training and validation accuracy during training.

Final performance should be documented after training and evaluation. For a more reliable estimate of generalization, evaluate the model on an independent test set that was not used for training or validation.

## 🚀 Future Improvements

- Experiment with transfer learning using EfficientNet or MobileNet.
- Improve generalization with an independent test set and diverse field images.
- Add prediction history and visual analytics.
- Deploy the application online.
- Explore explainability techniques such as Grad-CAM.

##  Disclaimer

This project is intended for educational and research purposes. Predictions are based on the trained model and dataset, and may be inaccurate on real-world images. Do not use predictions alone to make crop-treatment decisions.

##  Author

**Yashaswini Madhava**

GitHub: https://github.com/Yashaswinimadhava

---

⭐ If you find this project useful, consider starring the repository!
