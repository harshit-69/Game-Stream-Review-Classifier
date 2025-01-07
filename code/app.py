import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import pickle

def text_pre_process(text):
    non_punch=[c for c in text if c not in string.punctuation]
    non_punch="".join(non_punch)

    return[word for word in non_punch.split() if word.lower() not in stopwords.words("english")]










from flask import Flask, request, render_template
import pickle  # Use this if your model is saved as a pickle file

# Initialize Flask app
app = Flask(__name__)

# Load the trained review classifier model
model = pickle.load(open(r"C:\Users\Harshit\OneDrive\Desktop\GAME STREM REVIEW\code\review_model", "rb"))  # Replace 'review_model.pkl' with your file path

@app.route("/")
def home():
    return render_template("index.html")  # A simple form for input (create index.html)

@app.route("/predict", methods=["POST"])
def predict():
    # Get the review text from the form
    review = request.form["review"]
    prediction = model.predict([review])  # Pass the review directly to the model
    
    sentiment = "Positive" if prediction == "Positive" else "Negative"
    return render_template("index.html", review=review, sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
