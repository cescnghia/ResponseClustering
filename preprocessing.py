# -*- coding: utf-8 -*-

import nltk
from nltk import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import wordnet, stopwords
import re
import pandas as pd
import numpy as np


def standardize_text(df, text_field):
    
    """ 
        Input:
            - df: data's dataframe
            - text_field: column's (row of string) name that we want to standardize
        Output:
            - dataframe contains column `standardized`
    """
    
    urls_re = r'http\S+' #urls
    ht_re = r'#\S+'      #hashtags
    at_re = r'@\S+'      #@
    nb_re = " \d+"       #numbers

    punctuations = ["/","(",")","\\","|", ":",",",";","?", "!", "[", "]", "{","}"]
    
    new_text = []
    for text in df[text_field]:
        
        text = text.replace("\\n"," ").replace("\n", " ").replace("\\r"," ").replace("-"," ").replace("_"," ")
        
        for x in punctuations:
            text = text.replace(x , " ")
            
        text = re.sub(urls_re , " ", text)
        text = re.sub(ht_re   , " ", text)
        text = re.sub(at_re   , " ", text)
        text = re.sub(nb_re   , " ", text)
        
        text = text.replace("http"," ")
        
        new_text.append(text.lower())
    
    df['standardized_' + text_field] = new_text

    return df

def get_wordnet_pos(treebank_tag):
    """Map ['NN', 'NNS', 'NNP', 'NNPS'] to NOUN....."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def english_processing_sentence(sentence, tokenizer, lemmatiser, stop):
    "Tokenization"
    new_sentence = tokenizer.tokenize(sentence)
    "Lemmatization"
    tokens_pos = pos_tag(new_sentence)
    tokens_pos = [(w,get_wordnet_pos(p)) for (w,p) in tokens_pos]
    new_sentence = [lemmatiser.lemmatize(w, pos=p) for (w,p) in tokens_pos if p != None]
    "Stopwords removing"
    new_sentence = [x for x in new_sentence if x not in stop]

    return tokens_pos, new_sentence

def processing_text(df, text_field):
    
    """
        Input : 
            - Dataframe with the column's name (each row is a string) we want to process 
        Output:
            - One dataframe contains the processed original text
            - One dataframe in which each row is a 1-gram word (normalized word)
    """

    tokenizer = RegexpTokenizer(r'\w+')
    lemmatiser = WordNetLemmatizer()
    stop = stopwords.words('english')

    # Store processed original text
    new_text = []
    
    # Store normalized word (1-gram) and its parts of speech
    #original_words = []
    words = []
    pos = []
    
    for text in df[text_field]: # text may have multiple sentences
        new_sentences = ''
        for sentence in text.split('.'): # One sentence at a time (because we're playing with POS)
            
            tokens_pos, new_sent = english_processing_sentence(sentence, tokenizer, lemmatiser, stop)

            new_sentences += ' '.join(new_sent)
            new_sentences += ' '

            for (w, p) in tokens_pos:
                if (p == 'v' or p == 'n'):
                    word = lemmatiser.lemmatize(w, pos=p)
                    if word not in stop:
                        words.append(word)
                        pos.append(p)
                        #original_words.append(w)
        new_text.append(new_sentences)
    
    df['processed_text'] = new_text
    
    "Construct dataframe for each word, its count, its POS"
    df_normalised = pd.DataFrame({'processed_text': words, 'pos':pos})
    a = df_normalised.groupby(['processed_text']).agg('count')
    a = a.reset_index()
    a.columns = ['processed_text', 'count']
    b = df_normalised.drop_duplicates('processed_text')
    df_normalised = pd.merge(a, b, on='processed_text', how='outer')
    
    return df, df_normalised