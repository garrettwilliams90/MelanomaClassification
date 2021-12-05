# Create API of ML model using flask

'''
This code takes the JSON data while POST request an performs the prediction using loaded model and returns
the results in JSON format.
'''

# Import libraries
import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model 
from keras.preprocessing import image

app = Flask(__name__)

image_folder = os.path.join('static', 'images')
app.config["UPLOAD_FOLDER"] = image_folder

# Load the model
model = load_model('final_model.h5')

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict',methods=['POST'])
def predict():
    # Get the data from the POST request.
    if request.method == "POST":
        # predicting images
        imagefile = request.files['imagefile']
        image_path = './static/images/' + imagefile.filename 
        imagefile.save(image_path)

        img = image.load_img(image_path, target_size=(128, 128))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255.

        # Make prediction using model loaded from disk as per the data.
        prediction = model.predict(x)

        # Take the first value of prediction
        if prediction[0]>0.5:
            return render_template('result.html', output='Malignant! Please see your Dermatologist')
        else:
            return render_template('result.html', output='Benign. :)')

        return render_template("result.html", output=output)

if __name__ == '__main__':
    app.run(debug=True)