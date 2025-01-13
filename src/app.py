from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load('./models/symptom_checker_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.get('symptoms', '')
    if not data:
        return jsonify({'error': 'No symptoms provided'}), 400

    try:
        disease = model.predict([data])[0]
        return jsonify({'predicted_disease': disease})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)