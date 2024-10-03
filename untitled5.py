


from newsapi import NewsApiClient

import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

newsapi = NewsApiClient(api_key='f9be800d366e452f9616d299deb230f4')

def fetch_articles(query, category):
    articles = newsapi.get_everything(q=query, language='en', page_size=100)
    df = pd.DataFrame(articles['articles'])
    df['category'] = category
    return df

categories = ['tech', 'entertainment', 'business', 'sports', 'politics', 'travel', 'food', 'health']
dfs = []

for cat in categories:
    dfs.append(fetch_articles(cat, cat.capitalize()))

df = pd.concat(dfs)

def cleaned_desc_column(text):
    # Remove commas, extra spaces, full stops, quotes, and non-word characters
    text = re.sub(r'[,\.\'\"]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\W', ' ', text)

    # Tokenize and remove stopwords
    text_token = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_text = [word for word in text_token if word not in stop_words]

    return ' '.join(filtered_text)

import nltk
nltk.download('punkt')
nltk.download('stopwords') # Download the stopwords resource.
df['news_title'] = df['title'].apply(cleaned_desc_column)

X = df['news_title']
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=90)

lr = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000)),
])

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print(f"Accuracy is: {accuracy_score(y_test, y_pred)}")

news = [
    "Biden to Sign Executive Order That Aims to Make Child Care Cheaper",
    "Google Stock Loses $57 Billion Amid Microsoft's AI 'Lead'—And Reports It Could Be Replaced By Bing On Some Smartphones",
    "Poland suspends food imports from Ukraine to assist its farmers",
    "Can AI Solve The Air Traffic Control Problem? Let's Find Out",
    "Woman From Odisha Runs 42.5 KM In UK Marathon Wearing A Saree",
    "Hillary Clinton: Trump cannot win the election - but Biden will",
    "Jennifer Aniston and Adam Sandler starrer movie 'Murder Mystery 2' got released on March 24, this year"
]

predicted = lr.predict(news)

for doc, category in zip(news, predicted):
    print(f"'{doc}' => {category}")