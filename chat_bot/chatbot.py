#Import Packages
import json
import string
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from colorama import Fore,Back,Style
from keras.callbacks import EarlyStopping

#Load Dataset
with open("intents.json") as f:   #load json file containing intents
    data = json.load(f)

#Preprocess Data
lemmatizer = WordNetLemmatizer()       # Initializing lemmatizer to get stem of words​
words = []
classes = []
doc_X = []
doc_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)   # tokenize each patterns
        words.extend(tokens)                   # add tokens of words to list 'words'
        doc_X.append(pattern)                  # append patterns
        doc_y.append(intent["tag"])            # append tags

    # add the tag to the classes if it's not there already
    if intent["tag"] not in classes:
        classes.append(intent["tag"])
# lemmatize all the words and convert them to lowercase if the words don't appear in punctuation
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
words = sorted(set(words))            # remove duplicate words and sort them in alphabetical order
classes = sorted(set(classes))        # remove duplicate tags and sort them in alphabetical order

print("Length of doc_X :",len(doc_X))
print("Length of doc_X :",len(doc_y))
print("Number of unique words in patterns :",len(words))
print("Number of tag :",len(classes))
print("Types of Tag",classes)

#Create training data and labels
training = []                       # list for training data
out_empty = [0] * len(classes)      # list for labels
# creating the bag of words
for idx, doc in enumerate(doc_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())           # lematize and lowecase data
    for word in words:
        bow.append(1) if word in text else bow.append(0)
    output_row = list(out_empty)
    output_row[classes.index(doc_y[idx])] = 1   # mark the index of class that the current pattern is associated to
    training.append([bow, output_row])

random.shuffle(training)                     # shuffle the data
training = np.array(training, dtype=object)  # convert  to an array
# split the features and target labels
train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

#Build Model
input_shape = (len(train_X[0]),)       #shape of input data
output_shape = len(train_y[0])         #shape of output data
epochs = 200

model = Sequential()
model.add(Dense(128, input_shape=input_shape, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(output_shape, activation = "softmax"))
adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy',optimizer=adam, metrics=["accuracy"])
model.summary()

callback = EarlyStopping(monitor="loss", patience=7, restore_best_weights=True)
model.fit(x = train_X, y = train_y, epochs = 200, verbose = 1,callbacks = [callback])

#Cleaning text data
def clean_text(text):
    """
    Input:
    text = input text
    ___________________________________
    Output:
    tokens = token of words
    ____________________________________
    Objective : to clean the input text
    ____________________________________
    """
    tokens = nltk.word_tokenize(text)                            # tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens]     # lemmatize
    return tokens

def bag_of_words(text, vocab):
    """
    Input :
    text  =  input text
    vocab =  all unique words in the input data (patterns)
    ________________________________________________
    Output : returns bag of words
    _________________________________________________
    Objective : convert text into numeric value by checking the occurance of word in the text
    _________________________________________________
    """
    tokens = clean_text(text)        # clean_text function
    bow = [0] * len(vocab)           # empty bag of words with length = number of classes
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)

def pred_class(text, vocab, labels):
    """
    Input :
    text   = input text
    vocab  = unique words in input data (patterns)
    labels = different  classes of intents
    ________________________________________________
    Output : returns list of predicted intent classes
    _________________________________________________
    Objective : predicts the intent class using neural network
    __________________________________________________
    """
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]))[0]
    thresh = 0.2                                      # minimum threshold for probability
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]    #prediction greater than minimum threshold
    y_pred.sort(key=lambda x: x[1], reverse=True)     # sort probability in reverse order
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])              # takes first index of y_pred which has high probability
    return return_list

def get_response(intents_list, intents_json):
    """
    Input :
    intents_list = Predicted tag name
    intents_json = Input data
    ______________________________________________
    Output :
    result = randomly choosen response
    _______________________________________________
    Objective : get reponse for the predicted tag
    """
    tag = intents_list[0]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

#Running the Chatbot
print(Fore.RED + "Start messaging with the bot (type 'quit' to stop)!" + Style.RESET_ALL)
while True:
    print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
    message = input()
    if (message == "quit"):
        break

    intents = pred_class(message, words, classes)
    result = get_response(intents, data)
    print(Fore.GREEN + "Bot:" + Style.RESET_ALL ,result)
