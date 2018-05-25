# -*- coding: utf-8 -*-

import nltk
from nltk import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import wordnet, stopwords
import re
import pandas as pd
import numpy as np
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from nltk.tag import StanfordPOSTagger

POS_JAR = "/Users/Cescnghia/Documents/dataset/stanford-postagger-full-2017-06-09/stanford-postagger-3.8.0.jar"
POS_MODEL = "/Users/Cescnghia/Documents/dataset/stanford-postagger-full-2017-06-09/models/french.tagger"


def french_standardize_text(df, text_field):

    punctuations = ["/","(",")","\\","*","\'","&",">","=","≥","≤","|","\"", ":",",",";","?", "!", "[", "]", "{","}",'%','+','«','»']
    hours_re = "(?:\d{2}|\d{1})[a-zA-Z]" # remove 20h, 2h....
    nb_re = "\d+"                        #numbers
    
    new_text = []
    for text in df[text_field]:
        text = text.replace("\\n"," ").replace("\n", " ") \
                   .replace("\\r"," ").replace("-"," ") \
                   .replace("_"," ")

        text = re.sub(hours_re, " ", text)
        text = re.sub(nb_re   , " ", text)
        for x in punctuations:
            text = text.replace(x , " ")
        
        new_text.append(text.lower())
    
    df['standardized_' + text_field] = new_text

    return df


def get_french_pos(treebank_tag):
    """Map ['NN', 'NNS', 'NNP', 'NNPS'] to NOUN....."""
    if treebank_tag == 'ADJ':
        return wordnet.ADJ
    elif treebank_tag == 'ADV':
        return wordnet.ADV
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    else:
        return None

def french_processing_sentence(text, tokenizer, pos_tagger, lemmatizer, stop):
    "Tokenization"
    new_text = tokenizer.tokenize(text)
    "Lemmatization"
    tokens_pos = pos_tagger.tag(new_text)
    tokens_pos = [(w,get_french_pos(p)) for (w,p) in tokens_pos]
    new_text = [lemmatizer.lemmatize(w, pos=p) for (w,p) in tokens_pos if p != None]
    "Stopwords removing"
    new_text = [x for x in new_text if x not in stop]
    
    return new_text

def french_processing_text(df, text_field, POS_JAR, POS_MODEL):
    tokenizer = RegexpTokenizer(r'\w+')
    pos_tagger = StanfordPOSTagger(POS_MODEL, POS_JAR, encoding='utf8' )
    lemmatizer = FrenchLefffLemmatizer()
    stop = stopwords.words('french')

    # Store processed original dataset
    new_text = []
    
    # Store normalized words (1-gram) and its POS
    words = []
    pos = []

    for text in df[text_field]: # a text may have multiple sentences
        new_sentences = ''
        for sentence in text.split('.'): # Treat one sentence at a time. Why ? because we're playing with POS
            processed_sentence = french_processing_sentence(sentence, tokenizer, pos_tagger, lemmatizer, stop)
            new_sentences += ' '.join(processed_sentence) + ' '
        new_text.append(new_sentences)
        
    df['processed_text'] = new_text
    
    return df