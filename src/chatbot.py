import joblib
from nlp_pipeline import tokenize

def load_model(model_path):
    return joblib.load(model_path)

def predict_disease(model, symptoms):
    prediction = model.predict([symptoms])
    return prediction[0]

if __name__ == "__main__":
    model_path = '../models/symptom_checker_model.pkl'
    model = load_model(model_path)

    print("Welcome to the Symptom Checker Chatbot!")
    while True:
        user_input = input("Please describe your symptoms (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        disease = predict_disease(model, user_input)
        print(f"Based on the symptoms, you might have: {disease}")
