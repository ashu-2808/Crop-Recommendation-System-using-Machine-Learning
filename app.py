from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the trained model and scalers
model = pickle.load(open('model.pkl', 'rb'))
scaler_minmax = pickle.load(open('minmaxscaler.pkl', 'rb'))
scaler_std = pickle.load(open('standscaler.pkl', 'rb'))

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Get form data
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temperature = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    # Preprocess the data
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    scaled_features = scaler_minmax.transform(features)
    final_features = scaler_std.transform(scaled_features)

    # Make prediction
    prediction = model.predict(final_features)

    # Map prediction to crop
    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        recommended_crop = crop_dict[prediction[0]]
        result = f"{recommended_crop} is the best crop to be cultivated."
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
