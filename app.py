from flask import Flask, render_template, request
import joblib
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and the target encoder
try:
    model = joblib.load('mental_health_model.joblib')
    target_encoder = joblib.load('target_encoder.joblib')
except FileNotFoundError:
    print("Model files not found. Please run model_creation.py first.")
    exit()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    if request.method == 'POST':
        try:
            # Get data from the form
            form_data = request.form.to_dict()
            
            # Create a DataFrame with the correct column order for the model
            input_data = {
                'Age': [int(form_data['Age'])],
                'Gender': [form_data['Gender']],
                'family_history': [form_data['family_history']],
                'work_interfere': [form_data['work_interfere']],
                'benefits': [form_data['benefits']],
                'care_options': [form_data['care_options']],
                'anonymity': [form_data['anonymity']]
            }

            input_df = pd.DataFrame(input_data)
            
            # Make prediction
            prediction_encoded = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
            
            # Decode the prediction to 'Yes' or 'No'
            prediction = target_encoder.inverse_transform(prediction_encoded)[0]
            
            # Get the confidence score for the predicted class
            if prediction == 'Yes':
                confidence = prediction_proba[0][1] * 100
            else: # prediction == 'No'
                confidence = prediction_proba[0][0] * 100

            # Format the professional output text with HTML tags
            prediction_text = f"Predicted Outcome: <strong>{prediction}</strong>" \
                              f"<br>Confidence in this result: <strong>{confidence:.2f}%</strong>"

        except Exception as e:
            prediction_text = f"An error occurred: {e}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)