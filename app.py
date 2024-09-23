from flask import Flask, render_template, request
import numpy as np
import pickle
from PIL import Image
import io
from keras.models import load_model
#import keras
#import joblib

app = Flask(__name__)

# Load the model


model = load_model('model.h5')
#model = joblib.load("model.pkl")
#model = pickle.load(open("model.pkl", "rb"))

#model = load_model("model.pkl")
#model = keras.layers.TFSMLayer("model.pkl", call_endpoint='serving_default')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_form():
    try:
        # Log the received form data and files
        print("Form Data:", request.form)
        print("Files:", request.files)

        # Initialize variables
        numeric_inputs = None
        image = None

        # Check if numerical inputs are provided and valid
        try:
            input1 = float(request.form.get('input1', 'nan'))
            input2 = float(request.form.get('input2', 'nan'))
            input3 = float(request.form.get('input3', 'nan'))
            input4 = float(request.form.get('input4', 'nan'))
            input5 = float(request.form.get('input5', 'nan'))
            
            # Ensure that at least some valid numerical inputs are provided
            if not np.isnan([input1, input2, input3, input4, input5]).all():
                numeric_inputs = np.array([[input1, input2, input3, input4, input5]])
            
            print("Numeric inputs shape:", numeric_inputs.shape if numeric_inputs is not None else "None")
        except ValueError:
            # If conversion fails, numeric_inputs remain None
            print("No valid numerical inputs provided")

        # Check if the image file is provided
        image_file = request.files.get('imageUpload')

        if image_file and image_file.filename != '':
            try:
                image = Image.open(io.BytesIO(image_file.read()))
                image = image.convert('RGB')  # Ensure image is in RGB format
                # Resize the image to the model's expected input size (2048, 1024)
                #image = image.resize((2048, 1024))  # (width, height) as per the model requirement
                image = image.resize((64, 64))
                image = np.array(image) / 255.0   # Normalize to [0, 1] range
                image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 1024, 2048, 3)
                print("Image input shape:", image.shape)
            except Exception as e:
                print(f"Error opening image: {e}")
                return "Invalid image file", 400
        
        # Handle different cases based on the availability of inputs
        if numeric_inputs is not None and image is not None:
            # Both numerical and image data are provided
            result = model.predict([numeric_inputs, image])
        elif numeric_inputs is not None:
            # Only numerical data is provided
            # Use a placeholder for image input
            placeholder_image = np.zeros((1, 1024, 2048, 3))  # Placeholder for image input
             placeholder_image = np.zeros((1, 64, 64, 3))  # Placeholder for image input
            result = model.predict([numeric_inputs, placeholder_image])
        elif image is not None:
            # Only image data is provided
            # Use a placeholder for numerical input
            placeholder_numeric = np.zeros((1, 5))  # Placeholder for numerical input
            result = model.predict([placeholder_numeric, image])
        else:
            # No valid inputs provided
            return "Please provide at least one valid input (numeric or image).", 400

        # Print the prediction result to the terminal
        print("Prediction result array:", result)
        print(result[0][1])

        # Interpret the result
        if result[0][0] >=  0.5:
            pred_result = 'Leaf Spot Detected'
            fresult = 'Hence Unhealthy'
            prompt = "Summary: Leaf spot is a fungal disease causing dark, necrotic lesions on leaves."
            prompt2 = "Treatment: Remove and destroy infected leaves to prevent spread. Apply a fungicide specifically labeled for leaf spot control and ensure proper plant spacing for good air circulation."
        elif result[0][1] >=  0.02:
            pred_result = 'Anthracnose Detected'
            fresult = 'Hence Unhealthy'
            prompt = "Summary: Anthracnose is a fungal disease that causes dark, sunken lesions on leaves, stems, flowers, and fruit."
            prompt2 = "Treatment: Prune and discard affected plant parts and apply fungicides as recommended. Ensure plants are not overcrowded to reduce humidity levels that favor disease development."
        else:
            pred_result = 'Healthy'
            fresult = 'Hence Healthy'
            prompt = "Ensure plants receive adequate water, light, and nutrients, and maintain proper spacing to promote healthy growth and reduce the risk of disease. Regularly monitor plants for signs of pests and diseases and take prompt action to address any issues."
            prompt2 = None

        return render_template('output.html', result=pred_result, fresult=fresult, ai_summary=prompt, cure=prompt2)

    except Exception as e:
        print(f"Error occurred: {e}")
        return "An error occurred during prediction.", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
