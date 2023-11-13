import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from string import punctuation
import matplotlib.pyplot as plt
import seaborn as sns



def import_data():
    global true_data
    global fake_data

    true_data = pd.read_csv("../data/True.csv")
    fake_data = pd.read_csv("../data/Fake.csv")

def preprocess_data():
    global X
    global Y

    true_data["text"] = true_data["title"] + " - Subject: " + true_data["subject"] + " - " + true_data["text"]
    true_data["label"] = True
    fake_data["text"] = fake_data["title"] + " - Subject: " + fake_data["subject"] + " - " + fake_data["text"]
    fake_data["label"] = False

    data = pd.concat([true_data, fake_data])
    X = encode_with_word2vec(data["text"])
    Y = data["label"]


def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

def encode_with_word2vec(texts):
    sentences = [tokenize_and_lemmatize(text) for text in texts]
    vector_size = 100
    w2v_model = Word2Vec(sentences, vector_size=vector_size, window=5, min_count=10, workers=4)
    

    embedding_matrix = np.zeros((len(texts), vector_size))
    for i, text in enumerate(texts):
        avg_coord = np.zeros((vector_size))
        words = tokenize_and_lemmatize(text)
        
        present = 0
        for word in words:
            if word in w2v_model.wv:
                avg_coord += w2v_model.wv[word]
                present += 1
        avg_coord /= max(present, 1)  # Avoid division by zero
        
        embedding_matrix[i] = avg_coord

    return embedding_matrix

def shuffle_and_split_data():
    global X_train, X_validate, X_test, Y_train, Y_validate, Y_test
    data = pd.DataFrame(X)
    labels = pd.DataFrame(Y)

    X_train, X_temp, Y_train, Y_temp = train_test_split(data, labels, test_size=0.4, random_state=42)
    X_validate, X_test, Y_validate, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

def train_and_evaluate(k, X_train, Y_train, X_validate, Y_validate, X_test, Y_test):
    # Train a Naive Bayes classifier
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, Y_train.values.ravel())

    # Evaluate on the validation set
    Y_val_pred = classifier.predict(X_validate)
    val_accuracy = accuracy_score(Y_validate.values.ravel(), Y_val_pred)
    print("Validation Accuracy:", val_accuracy)

    # Evaluate on the test set
    Y_test_pred = classifier.predict(X_test)
    test_accuracy = accuracy_score(Y_test.values.ravel(), Y_test_pred)
    print("Test Accuracy:", test_accuracy)

    cm = confusion_matrix(Y_test, Y_test_pred)

    # Visualize the confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Real', 'Falsa'], yticklabels=['Real', 'Falsa'])
    plt.title('Matriz de Confusión - KNN y Word2Vec con k = ' + str(k) )
    plt.xlabel('Etiqueta Predicha')
    plt.ylabel('Etiqueta Verdadera')
    plt.show()

    return val_accuracy, test_accuracy


if __name__ == '__main__':
    np.random.seed(500)
    import_data()
    preprocess_data()
    shuffle_and_split_data()
    accuracy_scores = []
    for i in range(1, 10):
        k = 2*i - 1
        val_accuracy, test_accuracy = train_and_evaluate(k, X_train, Y_train, X_validate, Y_validate, X_test, Y_test)
        accuracy_scores.append([k, test_accuracy])
    
    # Plot test accuracy for different k values
    accuracy_scores = np.array(accuracy_scores)
    plt.plot(accuracy_scores[:, 0], accuracy_scores[:, 1])
    plt.title("Accuracy vs K")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.show()