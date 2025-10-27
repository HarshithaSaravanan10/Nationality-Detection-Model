🧠 Nationality, Age & Dress Colour Detection System
📘 Project Overview

This project predicts a person’s nationality, age, and dress colour from an uploaded image using a combination of Deep Learning and Computer Vision techniques.
It leverages DeepFace for accurate age estimation and a custom colour analysis algorithm for precise dress colour detection.

An elegant Streamlit interface enables users to upload images, preview results, and view detailed predictions instantly.

⚙️ Features

🧍 Detects faces automatically from uploaded images

🎂 Predicts age using the DeepFace model

👕 Identifies dress colour with accurate RGB-to-HSV analysis

🌎 Predicts nationality based on facial features

🖼️ Displays the original and processed images side-by-side

💻 Interactive Streamlit GUI for real-time analysis

🧠 Machine Learning Models Used

1️⃣ DeepFace (Age Prediction)

Model: DeepFace pre-trained age model

Framework: TensorFlow / Keras

Task: Estimate the person's age from facial features

2️⃣ CNN-based Nationality Classifier

Model Type: Convolutional Neural Network

Input: Face region

Output: Predicted nationality

3️⃣ Colour Detection Model

Technique: HSV-based pixel clustering

Task: Determine the dominant dress colour

🖥️ Streamlit Interface

Upload one or more images at once

View both original and annotated versions

Display detected attributes:

👶 Predicted Age

🌍 Predicted Nationality

🎨 Detected Dress Colour

🧰 Technologies Used

Python

OpenCV

NumPy

DeepFace

TensorFlow / Keras

Streamlit

Pillow

📦 Model Files

Since model files are large, you can download the pre-trained .h5 models directly from Google Drive:
👉 Download Model Files (Google Drive)

After downloading, place them inside a models directory in your project folder:

models/
├── nationality_model.h5
├── emotion_model.h5
├── hair_length_model.h5

🚀 How to Run

1️⃣ Clone the repository

git clone https://github.com/yourusername/Nationality-Age-Color-Detection.git
cd Nationality-Age-Color-Detection


2️⃣ Install dependencies

pip install -r requirements.txt


3️⃣ Download and add your .h5 model files

models/
├── nationality_model.h5
├── emotion_model.h5
├── hair_length_model.h5


4️⃣ Run the Streamlit app

streamlit run app.py

📊 Performance Highlights

✅ Works on static images and can be extended to live webcam feeds
✅ Uses pre-trained models for accurate predictions
✅ Provides clean visual outputs for each prediction

💬 Summary

The Nationality, Age & Dress Colour Detection System combines Computer Vision, Deep Learning, and Streamlit to deliver a smart, user-friendly application.
It detects faces, predicts nationality and age, and identifies the dominant dress colour with precision — a step toward real-world intelligent vision systems.
