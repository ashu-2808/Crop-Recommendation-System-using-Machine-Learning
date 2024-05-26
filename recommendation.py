import numpy as np
import pickle

# Load serialized objects
rfc = pickle.load(open('model.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))

# Define recommendation function
def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    # Transform features using loaded scalers
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    transformed_features = ms.transform(features)
    transformed_features = sc.transform(transformed_features)
    
    # Make prediction using loaded model
    prediction = rfc.predict(transformed_features).reshape(1, -1)
    
    return prediction[0]
