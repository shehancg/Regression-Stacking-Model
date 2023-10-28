import pickle
import pandas as pd
from flask import Flask, request, jsonify
import warnings

# Suppress warnings
# warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Load the stacking model from the pickle file
with open('stacking_model.pkl', 'rb') as file:
    stacking_model = pickle.load(file)

# Load the Excel file with BodyTypes and meals
meals_df = pd.read_excel('MealPlan.xlsx')  # Replace 'meals_data.xlsx' with the actual file path

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

        # Print the column names for reference
        print(meals_df.columns)
        
        # Retrieve meals for the BodyType from the DataFrame
        meals_data = meals_df[meals_df['BodyTypes'] == body_type]
        meals_list = meals_data.to_dict(orient='records')
        
        # Rearrange the keys for the result dictionary
        result = {
            "BMI": bmi,
            "Body Type": body_type,
            "Fat percentage": prediction,
            "Breakfast": meals_list[0]['Breakfast'],
            "Lunch": meals_list[0]['Lunch'],
            "Dinner": meals_list[0]['Dinner']
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
