# disease_predictor.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import difflib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (128, 128)
MODEL_PATH = "model/crop_disease_model.h5"
TRAIN_DIR = "PlantVillage"
REMEDY_CSV = "data/plant_disease_remedies_cleaned.csv"

# Load model and class labels once
model = load_model(MODEL_PATH)
print("✅ Model loaded")

datagen = ImageDataGenerator(rescale=1./255)
temp_gen = datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=1, class_mode='categorical')
class_labels = list(temp_gen.class_indices.keys())

remedy_df = pd.read_csv(REMEDY_CSV)
remedy_df["Disease"] = remedy_df["Disease"].str.strip().str.replace('"', '', regex=False)

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100
    predicted_class = class_labels[predicted_index].strip().replace('"', '')

    closest_match = difflib.get_close_matches(predicted_class, remedy_df["Disease"], n=1)
    if closest_match:
        remedy_row = remedy_df[remedy_df["Disease"] == closest_match[0]]
        remedy_text = remedy_row["Remedy"].values[0]
    else:
        remedy_text = "No remedy found in database."

    return predicted_class, confidence, remedy_text
