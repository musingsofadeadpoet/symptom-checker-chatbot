# Symptom Checker Chatbot 🩺

## Overview

Welcome to my first NLP attempt :) This is a Flask-based web application project that predicts possible diseases based on symptoms entered by the user. It leverages the Naive Bayes machine learning model to analyze user input (symptoms) and provide predictions. 

## Technology Stack

- Backend: Flask, Python

- Frontend: HTML, CSS, JavaScript

- Machine Learning: scikit-learn, CountVectorizer, Naive Bayes Classifier

- Data Processing: pandas

- Deployment: Local Flask server

## Installation and Setup

1. Clone the repository:

   git clone https://github.com/musingsofadeadpoet/symptom-checker-chatbot.git
   cd symptom-checker-chatbot

2. Set up a virtual environment

3. Install dependencies:

pip install -r requirements.txt

4. Preprocess the data: python src/data_preprocessing.py

5. Train the model: python src/train_model.py

6. Run the app: python src/app.py

8. Open the app in your browser to see the app live: http://127.0.0.1:5000

The app should look like this after you paste the local link to your browser:

   
<img width="1163" alt="Screenshot 2025-01-13 at 10 33 29 AM" src="https://github.com/user-attachments/assets/2556e4bd-85b1-43b0-aeef-7cf2bcd146e5" />



## Data

### Dataset Overview

Source: https://www.kaggle.com/datasets/choongqianzheng/disease-and-symptoms-dataset

### Data Preprocessing

The preprocessing script combines all symptoms into a single column and normalizes the text by converting it to lowercase. Missing values are removed, and the data is split into training and testing sets.

## Machine Learning Model
### Training Results

The model achieved an accuracy of 1.0 (100%) on the test dataset, as seen below:
<img width="920" alt="Screenshot 2025-01-13 at 10 31 20 AM" src="https://github.com/user-attachments/assets/f0f9f1ed-31c8-4455-9cee-0ae3789b95d7" />


### Training Code

The train_model.py script includes loading the processed training data, tokenizing and vectorizing symptoms using CountVectorizer, training a Naive Bayes classifier, and saving the trained model to models/symptom_checker_model.pkl.
