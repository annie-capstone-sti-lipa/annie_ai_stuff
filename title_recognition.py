from nltk.stem.lancaster import LancasterStemmer
import random
import tensorflow
import tflearn
import numpy
import json
import nltk


stemmer = LancasterStemmer()


with open("anime_titles.json") as file:
    titles = json.load(file)

print(titles)
