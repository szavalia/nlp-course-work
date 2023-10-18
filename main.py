import json

import pandas as pd
import nltk
from nltk.corpus import stopwords
from string import punctuation
from wordcloud import WordCloud
import matplotlib.pyplot as plt

true_normalized_tokens = []
fake_normalized_tokens = []
true_news = []
fake_news = []

def import_and_normalize_data():
    global true_normalized_tokens
    global fake_normalized_tokens
    global true_news
    global fake_news

    true = pd.read_csv("./corpus/True.csv")
    fake = pd.read_csv("./corpus/Fake.csv")

    corpus = {}

    true_news = true["text"].tolist()
    corpus["true_news"] = true_news
    fake_news = fake["text"].tolist()
    corpus["fake_news"] = true_news

    stops = stopwords.words('english')
    true_tokens = []
    fake_tokens = []

    for news in true_news:
        true_tokens += nltk.word_tokenize(news)
    corpus["true_tokens"] = true_tokens
    true_normalized_tokens = [token for token in true_tokens if token not in punctuation]
    true_normalized_tokens = [token for token in true_normalized_tokens if token not in stops]
    true_normalized_tokens = [word.lower() for word in true_normalized_tokens]
    corpus["true_normalized_tokens"] = true_normalized_tokens

    for news in fake_news:
        fake_tokens += nltk.word_tokenize(news)
    corpus["fake_tokens"] = fake_tokens
    fake_normalized_tokens = [token for token in fake_tokens if token not in punctuation]
    fake_normalized_tokens = [token for token in fake_normalized_tokens if token not in stops]
    fake_normalized_tokens = [word.lower() for word in fake_normalized_tokens]
    corpus["fake_normalized_tokens"] = fake_normalized_tokens

    # with open('corpus/corpus.json', 'w') as file:
    #    json.dump(corpus, file)

def get_saved_data():
    global true_normalized_tokens
    global fake_normalized_tokens
    global true_news
    global fake_news

    with open('corpus/corpus.json', 'r') as file:
        data = json.load(file)

    true_normalized_tokens = data["true_normalized_tokens"]
    fake_normalized_tokens = data["fake_normalized_tokens"]
    true_news = data["true_news"]
    fake_news = data["fake_news"]

def create_wordcloud():
    global true_normalized_tokens
    global fake_normalized_tokens

    true_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(true_normalized_tokens))
    plt.figure(figsize=(10, 5))
    plt.imshow(true_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    fake_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(fake_normalized_tokens))
    plt.figure(figsize=(10, 5))
    plt.imshow(fake_wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def get_avg_words_per_news():
    global true_normalized_tokens
    global fake_normalized_tokens
    global true_news
    global fake_news

    true_avg = len(true_normalized_tokens) / len(true_news)
    fake_avg = len(fake_normalized_tokens) / len(fake_news)

    plt.bar(["Real", "Falsa"], [true_avg, fake_avg])
    plt.ylabel("Promedio")
    plt.show()


import_and_normalize_data()
# get_saved_data()
create_wordcloud()
get_avg_words_per_news()