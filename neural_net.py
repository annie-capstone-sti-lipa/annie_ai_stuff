from nltk.stem.lancaster import LancasterStemmer
import random
import tensorflow
import tflearn
import numpy
import json
import nltk


stemmer = LancasterStemmer()


with open("intention.json") as file:
    dataset = json.load(file)

print(dataset)
