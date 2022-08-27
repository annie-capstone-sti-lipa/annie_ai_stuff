import pickle
import json
import random
import tensorflow
import tflearn
import numpy
import nltk

with open("anime_tits.json") as file:
    data = json.load(file)

with open("anime_titles.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)


net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 50)
net = tflearn.fully_connected(net, 50)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load("title_recognition.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [word.lower() for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print("Enter message (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        for text in ["my watchlist", "watchlist", "watch", "list", "add", "remove", "to my", "my", "to"]:
            inp = inp.replace(text, "")
        print(inp)

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)

        title = labels[results_index]
        title1 = labels[results_index+2]
        title2 = labels[results_index+1]

        print([title, title1, title2])


chat()
