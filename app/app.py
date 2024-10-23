# Import essential libraries
from flask import Flask, render_template, request, redirect
from markupsafe import Markup
import numpy as np
import pandas as pd
import requests
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
import config

# Initialize Flask app
app = Flask(__name__)

# Load plant disease classification model
disease_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
    'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]
disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu'), weights_only=True))
disease_model.eval()

# Load fertilizer recommendation data
df = pd.read_csv(r"C:\Users\kunal\OneDrive\Desktop\Harvestify-master\app\Data\fertilizer.csv")

# Custom functions
def weather_fetch(city_name):
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}appid={api_key}&q={city_name}"
    response = requests.get(complete_url).json()
    
    if response.get("cod") != "404":l;3
        
        main = response["main"]
        temperature = round((main["temp"] - 273.15), 2)
        humidity = main["humidity"]
        return temperature, humidity
    return None

def predict_image(img):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)
    
    with torch.no_grad():
        yb = disease_model(img_u)
    _, preds = torch.max(yb, dim=1)
    return disease_classes[preds[0].item()]

# Routes
@app.route('/')
def home():
    return render_template('index.html', title='Harvestify - Home')

@app.route('/fertilizer')
def fertilizer_recommendation():
    return render_template('fertilizer.html', title='Harvestify - Fertilizer Suggestion')

@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    crop_name = request.form['cropname']
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])

    crop_data = df[df['Crop'] == crop_name].iloc[0]
    nr, pr, kr = crop_data['N'], crop_data['P'], crop_data['K']
    
    diff = {'N': nr - N, 'P': pr - P, 'K': kr - K}
    max_diff = max(diff, key=lambda k: abs(diff[k]))
    key = f"{max_diff}High" if diff[max_diff] < 0 else f"{max_diff}low"

    response = Markup(str(fertilizer_dic.get(key, "No recommendation available")))
    return render_template('fertilizer-result.html', recommendation=response, title='Harvestify - Fertilizer Suggestion')

@app.route('/disease-predict', methods=['POST'])
def disease_prediction():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files.get('file')
    if file:
        try:
            img = file.read()
            prediction = predict_image(img)
            response = Markup(str(disease_dic.get(prediction, "No information available")))
            return render_template('disease-result.html', prediction=response, title='Harvestify - Disease Detection')
        except Exception as e:
            print(f"Error: {e}")
    return render_template('disease.html', title='Harvestify - Disease Detection')

if __name__ == '__main__':
    app.run(debug=False)
