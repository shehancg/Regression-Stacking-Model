import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the stacking model from the pickle file
with open('stacking_model.pkl', 'rb') as file:
    stacking_model = pickle.load(file)

def classify_body_type(bmi, fat_percentage):
    if bmi < 18.5 and fat_percentage < 20:
        return 'Ectomorph'
    elif 18.5 <= bmi < 24.9 and fat_percentage < 20:
        return 'Mesomorph'
    else:
        return 'Endomorph'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = [data.get('age'),
                      data.get('weight'), 
                      data.get('height'), 
                      data.get('neck'),
                      data.get('chest'), 
                      data.get('abdomen'), 
                      data.get('hip'), 
                      data.get('thigh'),
                      data.get('knee'), 
                      data.get('ankle'), 
                      data.get('biceps'), 
                      data.get('forearm'), 
                      data.get('wrist')]
        
        weight_lb = data.get('weight')
        height_in = data.get('height')

        # Convert height to meters (inches to meters)
        height_m = height_in * 0.0254

        # Calculate BMI using weight in kilograms and height in meters
        weight_kg = weight_lb * 0.45359237

        bmi = weight_kg / (height_m ** 2)

        prediction = stacking_model.predict([input_data])[0]
        body_type = classify_body_type(bmi, prediction)

        result = {'Fat percentage': prediction, 'BMI':bmi, 'Body Type': body_type}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
