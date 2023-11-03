import json

import numpy as np
import pandas as pd
import nltk
import torch
from nltk.corpus import stopwords
from string import punctuation

from torch import optim
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn as nn
import torch.nn.functional as functional

true_data = {}
fake_data = {}
train_data = {}
validate_data = {}
test_data = {}
model = None
tokenizer = None
criterion = None
optimizer = None


def import_data():
    global true_data
    global fake_data

    true_data = pd.read_csv("./corpus/True.csv")
    fake_data = pd.read_csv("./corpus/Fake.csv")


def normalize_data():
    global true_data
    global fake_data

    true_data['text'] = true_data.text.apply(normalize_string)
    fake_data['text'] = fake_data.text.apply(normalize_string)

    true_data['label'] = False
    fake_data['label'] = True


def normalize_string(s):
    stops = stopwords.words('english')

    aux = s.lower()
    aux = ''.join(x for x in aux if x not in punctuation)
    aux = ''.join(x for x in aux if x not in stops)
    return aux


def shuffle_and_split_data():
    global true_data
    global fake_data
    global train_data
    global validate_data
    global test_data

    all_data = pd.concat([true_data, fake_data])
    all_data = shuffle(all_data).reset_index(drop=True)
    train_data, validate_data, test_data = np.split(all_data.sample(frac=1),
                                                    [int(.6 * len(all_data)), int(.8 * len(all_data))])

    train_data = train_data.reset_index(drop=True)
    validate_data = validate_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)


def prepare_network():
    global tokenizer
    global model
    global criterion
    global optimizer

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    model.config.num_labels = 1

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(768, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
        nn.Softmax(dim=1)
    )

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.01)


def train_model():
    global model
    global train_data
    global optimizer
    global criterion

    print_every = 250

    total_loss = 0
    all_losses = []

    model.train()

    for idx, row in train_data.iterrows():
        text_parts = preprocess_text(str(row['text']))
        label = torch.tensor([row['label']]).long()

        optimizer.zero_grad()

        overall_output = torch.zeros((1, 2)).float()
        for part in text_parts:
            if len(part) > 0:
                part_input = part.reshape(-1)[:512].reshape(1, -1)
                overall_output += model(part_input, labels=label)[1].float()

        overall_output = functional.softmax(overall_output[0], dim=-1)

        if label == 0:
            label = torch.tensor([1.0, 0.0]).float()
        elif label == 1:
            label = torch.tensor([0.0, 1.0]).float()

        loss = criterion(overall_output, label)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        if idx % print_every == 0 and idx > 0:
            average_loss = total_loss / print_every
            print("{}/{}. Average loss: {}".format(idx, len(train_data), average_loss))
            all_losses.append(average_loss)
            total_loss = 0

    plt.plot(all_losses)
    plt.show()


def preprocess_text(text):
    global tokenizer

    parts = []

    text_len = len(text.split(' '))
    delta = 300
    max_parts = 5
    nb_cuts = int(text_len / delta)
    nb_cuts = min(nb_cuts, max_parts)

    for i in range(nb_cuts + 1):
        text_part = ' '.join(text.split(' ')[i * delta: (i + 1) * delta])
        parts.append(tokenizer.encode(text_part, return_tensors="pt", max_length=500))

    return parts


if __name__ == '__main__':
    import_data()
    normalize_data()
    shuffle_and_split_data()
    prepare_network()
    train_model()
