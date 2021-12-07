import streamlit as st
import sklearn
import keras.models import load_model
import keras.preprocessing import image
import numpy as np
<<<<<<< HEAD
=======
from flask import Flask, render_template, request, jsonify
import keras
from keras.models import load_model 
from keras.preprocessing import image
from PIL import Image
import os

app = Flask(__name__)

image_folder = os.path.join('static', 'images')
app.config["UPLOAD_FOLDER"] = image_folder
>>>>>>> e6ca36ffeb7183e939f9389d182886aed4b7fd7a

st.write("# Melanoma Classification")



uploaded_image = st.file_uploader("Choose a skin lesion image", type = "jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Skin Lesion', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    # Load the model
    model = load_model('final_model.h5')

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 128, 128, 3), dtype=np.float32)
    
    #image sizing
    size = (128, 128)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 255.0)

    # Load the image into the array
    data[0] = normalized_image_array

<<<<<<< HEAD
    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability
    if prediction == 0:
        st.write("The skin lesion is Benign")
    else:
        st.write("The skin lesion is Malignant. Please see a Dermatologist")
=======
if __name__ == '__main__':
    app.run(debug=True)
>>>>>>> e6ca36ffeb7183e939f9389d182886aed4b7fd7a
