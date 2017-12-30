#!/usr/bin/env python

'''
cs 375
hw 1
tongyu zhou
'''

import math
import operator
import random
import string
import re
from collections import Counter

# Stores n-gram counts for any n for any given corpus
def ngram_count(x, text):
    ngrams = [' '.join(text[i: i+x]) for i in range(len(text) - x+1)]
    return dict(Counter(ngrams))

# Stores n-gram log probabilities for any n for any given corpus
# token_count = N
# type_count = V
def ngram_log_prob(x, text):
    log_probabilities = {}
    numerators = ngram_count(x, text)
    if x > 1: antecedents = ngram_count(x-1, text)
    type_count = len(set(text))
    for key in list(numerators.keys()):
        if x > 1: token_count = antecedents[' '.join(key.split()[:x-1])]
        else: token_count = len(text)
        # add-1 smoothing
        log_probabilities[key] = math.log((numerators[key])/ float(token_count))
    return log_probabilities

# Prints most frequent n-grams
def find_count(x, text, slice_num):
    counts = ngram_count(x, text)
    sliced = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)[:slice_num]
    for i in sliced:
        print(list(i)[0] + ': ' + str(list(i)[1])) # add-1 smoothing

# Prints log-probabilities
def find_prob(x, text, slice_num):
    probabilities = ngram_log_prob(x, text)
    sliced = sorted(probabilities.items(), key=operator.itemgetter(1), reverse=True)[:slice_num]
    for i in sliced:
        print(list(i)[0] + ': ' + str(list(i)[1]))

# Generates the next word based on previous n-1 words
def next_word(x, token, dict, text):
    if x == 1: return random.choice(list(dict.keys()))
    next = []
    for k,v in dict.items():
        if k.startswith(token): next += v * [k.split()[-1]] 
    next.extend(text) # add-1 smoothing for words not in corpus
    return random.choice(text)

# Generates a sentence with n-grams
def generate_sent(x, text):
    sentence = []
    counts = ngram_count(x, text)
    next = '<s>' # <s> is first token of each sentence
    while '</s>' not in next and len(sentence) < 20: # threshold = 100 tokens
        sentence.append(next)
        next = next_word(x, (' '.join(sentence[-(x-1):])), counts, text)
    while '<s>' in sentence: sentence.remove('<s>')
    return str(' '.join(sentence))

########################################################
# uncomment out each section to run 

#replace nltk tokenizer with regular expression tokenizer
corpus = []
import nltk
def preprocess():
    sentences = nltk.sent_tokenize(open('category_corpus.txt', 'r').read())
    for i in sentences:
        corpus.append("<s>")
        corpus.extend(re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", i))
        corpus.append("</s>")
    return corpus

