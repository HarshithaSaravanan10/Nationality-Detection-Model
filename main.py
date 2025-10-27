import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from deepface import DeepFace
from PIL import Image

# --------------------------
# Load models
# --------------------------
nationality_model = load_model('nationality_model.h5')
emotion_model = load_model('emotion_model.h5') 
age_model = load_model('age_prediction_model.h5') # your CNN model

# --------------------------
# Helper functions
# --------------------------

# Nationality prediction (7-class mapping)
def predict_nationality(img):
    img_resized = cv2.resize(img, (128,128))
    img_array = img_resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = nationality_model.predict(img_array)
    
    nationalities = [
        'African', 
        'East Asian', 
        'Indian', 
        'Latino_Hispanic', 
        'Middle Eastern', 
        'Southeast Asian', 
        'United Nation'
    ]
    
    return nationalities[np.argmax(pred)]

# Emotion prediction (7-class mapping)
def predict_emotion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, (48,48))
    img_array = gray_resized.reshape(1,48,48,1)/255.0
    pred = emotion_model.predict(img_array)
    
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    return emotions[np.argmax(pred)]

# Age prediction using DeepFace (safe handling)
def predict_age(image_path):
    try:
        result = DeepFace.analyze(img_path=image_path, actions=['age'], enforce_detection=False)
        if isinstance(result, list):
            result = result[0]
        return result['age']
    except Exception as e:
        return "Age not detected"

# Dress color detection using OpenCV (dominant color)
def detect_dress_color_opencv(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "Image not found"
    
    img = cv2.resize(img, (400,400))
    h, w, _ = img.shape
    y1, y2 = int(h*0.4), int(h*0.8)
    torso = img[y1:y2, :]
    
    torso_rgb = cv2.cvtColor(torso, cv2.COLOR_BGR2RGB)
    pixels = torso_rgb.reshape(-1,3)
    
    mask = np.all((pixels > [20,20,20]) & (pixels < [235,235,235]), axis=1)
    pixels = pixels[mask]
    
    if len(pixels) == 0:
        pixels = torso_rgb.reshape(-1,3)
    
    avg_color = np.mean(pixels, axis=0).astype(int)
    
    BASIC_COLORS = {
        'red': (255,0,0),
        'green': (0,128,0),
        'blue': (0,0,255),
        'yellow': (255,255,0),
        'orange': (255,165,0),
        'pink': (255,192,203),
        'purple': (128,0,128),
        'brown': (165,42,42),
        'black': (0,0,0),
        'white': (255,255,255),
        'gray': (128,128,128),
        'khaki': (195,176,145)
    }
    
    def closest_color(avg_color):
        min_dist = float('inf')
        closest = None
        for name, rgb in BASIC_COLORS.items():
            dist = np.sum((np.array(rgb)-avg_color)**2)
            if dist < min_dist:
                min_dist = dist
                closest = name
        return closest
    
    color_name = closest_color(avg_color)
    return color_name

# --------------------------
# Streamlit GUI
# --------------------------
st.title("🌍 Nationality & Attribute Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image_np = np.array(image.convert('RGB'))
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Save temporarily for DeepFace
    temp_path = "temp_image.jpg"
    image.save(temp_path)

    # Step 1: Nationality
    nationality = predict_nationality(image_cv)
    st.write("**Nationality:**", nationality)

    # Step 2: Emotion
    emotion = predict_emotion(image_cv)
    st.write("**Emotion:**", emotion)

    # Step 3: Conditional predictions
    age = None
    dress_color = None

    if nationality == "Indian":
        age = predict_age(temp_path)
        dress_color = detect_dress_color_opencv(temp_path)
    elif nationality == "United Nation":  # treat as USA-like
        age = predict_age(temp_path)
    elif nationality == "African":
        dress_color = detect_dress_color_opencv(temp_path)

    if age:
        st.write("**Age:**", age)
    if dress_color:
        st.write("**Dress Color:**", dress_color)
