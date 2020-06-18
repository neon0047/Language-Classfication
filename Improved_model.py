import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import string

from collections import defaultdict
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

import joblib
import pickle as pkl
from helper_code import *  


def open_text_file(filename):
    with open(filename,"r", encoding="utf-8") as f:
        data = f.readlines()
    return data   

raw_data = dict()
raw_data['sk'] = open_text_file('Sentences/train_sentences.sk')
raw_data['cs'] = open_text_file('Sentences/train_sentences.cs')
raw_data['en'] = open_text_file('Sentences/train_sentences.en')


def show_statistic(data):
    for languages, sentences in data.items():
        
        no_of_sentences = 0
        no_of_words = 0
        no_of_unique_words = 0
        sample_extract = ""
        
        word_list = ' '.join(sentences).split()
        
        no_of_sentences = len(sentences)
        no_of_words = len(word_list)
        no_of_unique_words = len(set(word_list))
        sample_extract = " ".join(sentences[0].split()[0:7])
  
        print(f'Language:{languages}')
        print('-----------------------')
        print(f'Number of sentences\t:\t  {no_of_sentences}')
        print(f'Number of words\t\t:\t  {no_of_words}')
        print(f'Number of unique words\t:\t  {no_of_unique_words}')
        print(f'Sample extract\t\t:\t  {sample_extract}....\n')
        
        
#show_statistic(raw_data)  
#do_law_of_zipf(raw_data)     


#Data Cleaning and preprocessing
def preprocess(text):
    preprocessed_text = text.lower().replace('-',' ')
    translation_table = str.maketrans('\n', ' ', string.punctuation+string.digits)
    preprocessed_text = preprocessed_text.translate(translation_table)
    
    return preprocessed_text

data_preprocessed = {k: [preprocess(sentence) for sentence in v] for k, v in raw_data.items()}

'''print('Raw')
show_statistic(raw_data)
print('\n preprocessed')
show_statistic(data_preprocessed)'''

#vectorizing Training Data
sentence_train , y_train = [], []
for k, v in data_preprocessed.items():
    for sentence in v:
        sentence_train.append(sentence)
        y_train.append(k)


vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(sentence_train)

#Initializing Model Parameters and training
naive_classifier = MultinomialNB()
naive_classifier.fit(X_train, y_train)    

#Vectorizing Validation Data and Evaluation Model
data_val = dict()
data_val['sk'] = open_text_file('sentences/val_sentences.sk')
data_val['cs'] = open_text_file('sentences/val_sentences.cs')  
data_val['en'] = open_text_file('sentences/val_sentences.en')

data_val_preprocessed = {k:[preprocess(sentence) for sentence in v] for k, v in data_val.items()}

sentence_val, y_val = [],[]
for k, v in data_val_preprocessed.items():
    for sentence in v:
        sentence_val.append(sentence)
        y_val.append(k)


X_val = vectorizer.transform(sentence_val)

'''prediction = naive_classifier.predict(X_val)

plot_confusion_matrix(y_val, prediction, ['sk','cs','en'])'''   


#simple adjustments and highlighting model shortcomings
naive_classifier = MultinomialNB(alpha = 0.0001, fit_prior=False)
naive_classifier.fit(X_train, y_train)

prediction = naive_classifier.predict(X_val)

plot_confusion_matrix(y_val, prediction, ['sk','cs','en'])   

f1_score(y_val, prediction , average = 'weighted')

 # taken from https://arxiv.org/abs/1508.07909
    
# taken from https://arxiv.org/abs/1508.07909

import re, collections
def get_stats(vocab):
    pairs = collections.defaultdict(int) 
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq 
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word] 
    return v_out

def get_vocab(data):

    words = []
    for sentence in data:
        words.extend(sentence.split())
        
    vocab = defaultdict(int)
    for word in words:
        vocab[' '.join(word)] += 1
        
    return vocab

vocab = get_vocab(sentence_train)

# also taken from original paper
for i in range(100):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get) 
    vocab = merge_vocab(best, vocab)
    
merges = defaultdict(int)
for k, v in vocab.items():
    for subword in k.split():
        if len(subword) >= 2:
            merges[subword] += v
            
            
merge_ordered = sorted(merges, key=merges.get, reverse=True)        

pkl.dump(merge_ordered, open('Data/Auxiliary/merge_ordered.pkl', 'wb'))  

def split_into_subwords(text):
    merges = pkl.load(open('Data/Auxiliary/merge_ordered.pkl', 'rb'))
    subwords = []
    for word in text.split():
        for subword in merges:
            subword_count = word.count(subword)
            if subword_count > 0:
                word = word.replace(subword, ' ')
                subwords.extend([subword]*subword_count)
    return ' '.join(subwords)  

split_into_subwords('this is ari here')

data_preprocessed_subwords = {k: [split_into_subwords(sentence) for sentence in v] for k, v in data_preprocessed.items()}
            
show_statistic(data_preprocessed_subwords)

data_train_subwords = []
for sentence in sentence_train:
    data_train_subwords.append(split_into_subwords(sentence))          
            
data_val_subwords = []
for sentence in sentence_val:
    data_val_subwords.append(split_into_subwords(sentence))   



vectorizer = CountVectorizer()

X_train = vectorizer.fit_transform(data_train_subwords)
X_val = vectorizer.transform(data_val_subwords)

naive_classifier = MultinomialNB(fit_prior=False)
naive_classifier.fit(X_train, y_train)


predictions = naive_classifier.predict(X_val)

plot_confusion_matrix(y_val, predictions, ['sk', 'cs', 'en'])

f1_score(y_val, predictions, average='weighted')

        
