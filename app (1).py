import re
import os
import requests
import numpy as np
from bs4 import BeautifulSoup
import urllib.request as urllib
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import time
import pandas as pd
import csv
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to read CSV file
def read_csv(file_path):
    reviews = []
    sentiments = []

    try:
        with open(file_path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)

            for row in csv_reader:
                if len(row) == 2:
                    review_text, sentiment = row
                    reviews.append(review_text)
                    sentiments.append(sentiment)
                else:
                    print(f"Skipping row: {row}. It does not have two values.")
    except FileNotFoundError:
        print(f"File not found at path: {file_path}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")

    return reviews, sentiments

# Load CSV file for training
train_file_path = 'amazon_review - Copy.csv'  # Change this to the path of your CSV file
train_reviews, train_sentiments = read_csv(train_file_path)

# Function to clean text
def clean(x):
    x = re.sub(r'[^a-zA-Z ]', ' ', x)  # Replace everything that's not an alphabet with a space
    x = re.sub(r'\s+', ' ', x)  # Replace multiple spaces with one space
    x = re.sub(r'READ MORE', '', x)  # Remove READ MORE
    x = x.lower()
    x = x.split()
    y = []
    for i in x:
        if len(i) >= 3:
            if i == 'osm':
                y.append('awesome')
            elif i == 'nyc':
                y.append('nice')
            elif i == 'thanku':
                y.append('thanks')
            elif i == 'superb':
                y.append('super')
            else:
                y.append(i)
    return ' '.join(y)

# Function to predict sentiment
def predict_sentiment(review):
    inputs = tokenizer(review, return_tensors="pt", truncation=True)
    inputs.to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    sentiment = "POSITIVE" if predicted_class == 1 else "NEGATIVE"
    return sentiment, probabilities[0][predicted_class].item()

# Your function to process reviews and obtain positive and negative reviews
def process_reviews(product_url):
    positive_reviews = []
    negative_reviews = []

    # Perform web scraping or any other method to obtain reviews
    # Replace the following with your logic to get positive and negative reviews
    # For example, you can use sentiment analysis to classify reviews

    # Sample logic (replace it with your actual logic)
    sample_reviews = [
        {"text": "Positive review 1", "sentiment": "positive"},
        {"text": "Negative review 1", "sentiment": "negative"},
        # Add more reviews as needed
    ]

    for review in sample_reviews:
        if review["sentiment"] == "positive":
            positive_reviews.append(review["text"])
        elif review["sentiment"] == "negative":
            negative_reviews.append(review["text"])

    return positive_reviews, negative_reviews

# Function to extract all reviews
def extract_all_reviews(url, clean_reviews, org_reviews, customernames, commentheads, ratings):
    with urllib.urlopen(url) as u:
        page = u.read()
        page_html = BeautifulSoup(page, "html.parser")
    reviews = page_html.find_all('div', {'class': 't-ZTKy'})
    commentheads_ = page_html.find_all('p', {'class': '_2-N8zT'})
    customernames_ = page_html.find_all('p', {'class': '_2sc7ZR _2V5EHH'})
    ratings_ = page_html.find_all('div', {'class': ['_3LWZlK _1BLPMq', '_3LWZlK _32lA32 _1BLPMq', '_3LWZlK _1rdVr6 _1BLPMq']})

    for review in reviews:
        x = review.get_text()
        org_reviews.append(re.sub(r'READ MORE', '', x))
        clean_reviews.append(clean(x))

    for cn in customernames_:
        customernames.append('~' + cn.get_text())

    for ch in commentheads_:
        commentheads.append(ch.get_text())

    ra = []
    for r in ratings_:
        try:
            if int(r.get_text()) in [1, 2, 3, 4, 5]:
                ra.append(int(r.get_text()))
            else:
                ra.append(0)
        except:
            ra.append(r.get_text())

    ratings += ra
    print(ratings)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['GET'])
def result():
    url = request.args.get('url')
    nreviews = int(request.args.get('num'))
    clean_reviews = []
    org_reviews = []
    customernames = []
    commentheads = []
    ratings = []

    with urllib.urlopen(url) as u:
        page = u.read()
        page_html = BeautifulSoup(page, "html.parser")

    proname = page_html.find_all('span', {'class': 'B_NuCI'})[0].get_text()
    price = page_html.find_all('div', {'class': '_30jeq3 _16Jk6d'})[0].get_text()

    # getting the link of see all reviews button
    all_reviews_url = page_html.find_all('div', {'class': 'col JOpGWq'})[0]
    all_reviews_url = all_reviews_url.find_all('a')[-1]
    all_reviews_url = 'https://www.flipkart.com' + all_reviews_url.get('href')
    url2 = all_reviews_url + '&page=1'

    # start reading reviews and go to the next page after all reviews are read
    while True:
        x = len(clean_reviews)
        # extracting the reviews
        extract_all_reviews(url2, clean_reviews, org_reviews, customernames, commentheads, ratings)
        url2 = url2[:-1] + str(int(url2[-1]) + 1)
        if x == len(clean_reviews) or len(clean_reviews) >= nreviews:
            break

    org_reviews = org_reviews[:nreviews]
    clean_reviews = clean_reviews[:nreviews]
    customernames = customernames[:nreviews]
    commentheads = commentheads[:nreviews]
    ratings = ratings[:nreviews]

    # Your existing code for word cloud generation

    # making a dictionary of product attributes and saving all the products in a list
    d = []
    for i in range(len(org_reviews)):
        x = {}
        x['review'] = org_reviews[i]
        x['cn'] = customernames[i]
        x['ch'] = commentheads[i]
        x['stars'] = ratings[i]
        d.append(x)

    for i in d:
        if i['stars'] != 0:
            sentiment, probability = predict_sentiment(i['review'])
            i['sent'] = sentiment
            i['probability'] = probability

            if sentiment == 'POSITIVE' and probability >= 0.7:
                i['sent'] = 'UNCERTAIN_POSITIVE'
            elif sentiment == 'NEGATIVE' and probability >= 0.7:
                i['sent'] = 'UNCERTAIN_NEGATIVE'

    np, nn = 0, 0
    for i in d:
        if i['sent'] == 'POSITIVE':
            np += 1
        elif i['sent'] == 'NEGATIVE':
            nn += 1

    return render_template('results.html', dic=d, n=len(clean_reviews), nn=nn, np=np, proname=proname, price=price)

@app.route('/analyze_reviews', methods=['POST'])
def analyze_reviews():
    # Process the product URL and obtain positive and negative reviews
    product_url = request.form['product_url']
    positive_reviews, negative_reviews = process_reviews(product_url)

    return render_template('result.html', positive_reviews=positive_reviews, negative_reviews=negative_reviews)

@app.route('/wc')
def wc():
    return render_template('wc.html')


class CleanCache:
    def __init__(self, directory=None):
        self.clean_path = directory
        if os.listdir(self.clean_path) != list():
            files = os.listdir(self.clean_path)
            for fileName in files:
                os.remove(os.path.join(self.clean_path, fileName))
        print("Cleaned!")

if __name__ == '__main__':
    app.run(debug=True)
