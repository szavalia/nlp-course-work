import json

import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
from string import punctuation

from torch import optim
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
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

fake_random = """
Americans to fund killing babies in abortion that she has been caught trying to add taxpayer financing of abortions to the bill to combat the Coronavirus and provide economic stimulus to the nation as it deals with the COVD-19 outbreak.
Nancy Pelosi has a long history of promoting abortion and her first act after becoming Speaker in 2019 was pushing legislation to use tax money for abortions. So it’s no surprise she is trying to exploit the Coronavirus pandemic to push abortion funding again.
As The Daily Caller reports: House Speaker Nancy Pelosi sought to include a potential way to guarantee federal funding for abortion into the coronavirus economic stimulus plan, according to multiple senior White House officials.
Speaking to the Daily Caller, those officials alleged that while negotiating the stimulus with U.S. Treasury Secretary Steve Mnuchin, Pelosi tried to lobby for “several” provisions that stalled bipartisan commitment to the effort. One was a mandate for up to $1 billion to reimburse laboratory claims, which White House officials say would set a precedent of health spending without protections outlined in the Hyde Amendment.
LifeNews depends on the support of readers like you to combat the pro-abortion media. Please donate now.
“A New mandatory funding stream that does not have Hyde protections would be unprecedented,” one White House official explained. “Under the guise of protecting people, Speaker Pelosi is working to make sure taxpayer dollars are spent covering abortion—which is not only backwards, but goes against historical norms.”
A second White House official referred to the provision as a “slush fund” and yet another questioned “what the Hyde Amendment and abortion have to do with protecting Americans from coronavirus?”
Americans should insist to their members of Congress that we need a clean bill that provides aggressive action to help patients and spur the economy. Killing babies with our tax dollars is not the answer to the coronavirus and the situation should not be exploited for political gain.
"""

real_random = """
Price spikes, however, would cause demand to wither and some expensive avocados might be leftover, and stores might try to ration avocados, he added.
"Exactly what the retail strategy would be in this case, I’m not sure. But we would have vastly fewer avocados," Sumner said.
Just how fast avocados would disappear, if at all, would depend on whether the Trump administration enacts a full or partial border closure. White House economic adviser Larry Kudlow told CNBC he’s looking for ways to keep some commerce flowing.
"We are looking at different options, particularly if you can keep those freight lanes, the truck lanes, open," he said this week.  
Ben Holtz owns Rocky H Ranch, a 70-acre family-run avocado farm in northern San Diego County. He agreed avocados would run out within weeks.
"Mexico is the big player today. California is not. You shut down the border and California can’t produce to meet the demand," Holtz said. "There will be people without their guacamole."
While Mexico’s avocado harvest is year-round, California’s is limited to April through July. Growers in the state have picked only about 3 percent of what’s expected to be a much smaller crop of about 175 million pounds this year, Holtz said. A heat wave last summer reduced the crop size.
California’s avocado harvest has averaged approximately 300 million pounds in recent years, according to data from the California Avocado Commission. By contrast, the U.S. has imported more than 1.5 billion pounds of avocados from Mexico annually. Representatives from the commission did not respond to requests for this article.
Altogether, the U.S. received 43 percent of its fruit and vegetable imports from Mexico in 2016, according to the U.S. Department of Agriculture.
Also affecting this year’s avocado supply, a California avocado company in March recalled shipments to six states last month after fears the fruit might be contaminated with a bacterium that can cause health risks.
Until the early 2000s, California was the nation’s leading supplier of avocados, Holtz said. Mexico gradually overtook the state and now dominates sales in the U.S.
"It’s a very big possibility," Holtz said of avocado shortages. "Three weeks would dry up the Mexican inventory. California alone consumes more avocados than are grown in our state. Cold storage supply chain is basically three weeks or less of inventory. Most of the time it’s seven days."
A spokeswoman for the California Restaurant Association said "we haven’t heard concerns from restaurants, it doesn’t mean they aren’t worried." A national grocers association said it will "continue to closely monitor any developments" at the border, but did not have information about the potential impact on avocados.
"""

def import_data():
    global true_data
    global fake_data

    true_data = pd.read_csv("./data/True.csv")
    fake_data = pd.read_csv("./data/Fake.csv")

def normalize_data():
    global true_data
    global fake_data

    true_data["text"] = true_data["title"] + " - Subject: " + true_data["subject"] + " - " + true_data["text"]
    fake_data["text"] = fake_data["title"] + " - Subject: " + fake_data["subject"] + " - " + fake_data["text"]

    true_data['label'] = False
    fake_data['label'] = True

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

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
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


def validate_model():
    global model
    global validate_data
    global optimizer
    global criterion

    print_every = 250

    total_loss = 0
    all_losses = []

    model.eval()

    with torch.no_grad():
        for idx, row in validate_data.iterrows():
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

            if idx % print_every == 0 and idx > 0:
                average_loss = total_loss / print_every
                print("{}/{}. Average loss: {}".format(idx, len(validate_data), average_loss))
                all_losses.append(average_loss)
                total_loss = 0

    plt.plot(all_losses)
    plt.show()


def test_model():
    global model
    global test_data

    total = len(test_data)
    numbers_right = [0, 0, 0]  # Real, Fake, All
    numbers_by_label = [0, 0]  # Real, Fake
    print_every = 250
    all_accuracies = []

    model.eval()
    with torch.no_grad():
        for idx, row in test_data.iterrows():
            text_parts = preprocess_text(str(row['text']))
            label = torch.tensor([row['label']]).float()

            overall_output = torch.zeros((1, 2))
            for part in text_parts:
                if len(part) > 0:
                    overall_output += model(part.reshape(1, -1))[0]

            overall_output = functional.softmax(overall_output[0], dim=-1)

            result = overall_output.max(0)[1].float().item()

            if row['label']:
                numbers_by_label[1] += 1
            else:
                numbers_by_label[0] += 1

            if result == label.item():
                numbers_right[2] += 1
                if row['label']:
                    numbers_right[1] += 1
                else:
                    numbers_right[0] += 1

            if idx % print_every == 0 and idx > 0:
                accuracy = numbers_right[2] / idx
                print("{}/{}. Current accuracy: {}".format(idx, total, accuracy))
                all_accuracies.append(accuracy)

    print("Accuracy on test data: {}".format(numbers_right[2] / total))
    print("Total: {}, Numbers Right: {}".format(total, numbers_right[2]))
    print("Fake total: {}, Fake right: {}".format(numbers_by_label[1], numbers_right[1]))
    print("True total: {}, True right: {}".format(numbers_by_label[0], numbers_right[0]))
    plt.plot(all_accuracies)
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


def test_random(txt):
    text_parts = preprocess_text(txt)
    overall_output = torch.zeros((1, 2))
    for part in text_parts:
        if len(part) > 0:
            overall_output += model(part.reshape(1, -1))[0]

    overall_output = functional.softmax(overall_output[0], dim=-1)

    value, result = overall_output.max(0)

    term = "fake"
    if result.item() == 0:
        term = "real"

    print("{} at {}%".format(term, value.item() * 100))


if __name__ == '__main__':
    import_data()
    normalize_data()
    shuffle_and_split_data()
    prepare_network()
    train_model()
    validate_model()
    test_model()
    test_random(fake_random)
    test_random(real_random)