import streamlit as st
import sklearn
import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image

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

    # run the inference
    prediction = model.predict(data)
    prediction = np.argmax(prediction) # return position of the highest probability
    if prediction == 0:
        st.write("The skin lesion is Benign")
    else:
        st.write("The skin lesion is Malignant. Please see a Dermatologist")

