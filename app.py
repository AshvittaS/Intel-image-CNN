import gradio as gr
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import pickle

# Load model and class names
model = load_model('intel_model.h5')
with open('class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)

IMAGE_SIZE = (150, 150)

def predict_image(img):
    # Convert to numpy
    img = cv2.resize(img, IMAGE_SIZE)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    pred = model.predict(img)
    index = np.argmax(pred)
    return {class_names[index]: float(pred[0][index])}

gr.Interface(fn=predict_image,
             inputs=gr.Image(type="numpy"),
             outputs=gr.Label(num_top_classes=3),
             title="Intel Image Classifier").launch()
