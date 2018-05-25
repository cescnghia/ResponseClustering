# -*- coding: utf-8 -*-

import nltk
from nltk import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import wordnet, stopwords
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from concept import find_hypernym

def find_replacement(df, threshold):
    new_df = df.copy()
    words = np.array(df['processed_text'])
    pos   = np.array(df['pos'])
    X = [x.tolist() for x in df['vect']]
    sim_matrix = cosine_similarity(X, X)
    new_words = words.copy()
    to_replace = []
    replaced_by = []
    for idx, word in enumerate(words):
        sim = sim_matrix[idx,:]
        max_sim = np.where(sim > threshold)[0]
        
        top_words = words[max_sim]
        top_pos   = pos[max_sim]
        top_scores= sim[max_sim]

        if (len(top_words) > 1):
            #print(top_words)
            #print(top_pos)
            #print(top_scores)
            concept = find_hypernym(top_words, top_pos, 3)
            if concept == 'Not Found' or concept == '*ROOT*':
                replace_ind = top_scores.argsort()[::-1][1]
                concept = top_words[replace_ind]
            to_replace.append(top_words)
            replaced_by.append(concept)
            new_words[max_sim] = concept
            #print('We can actually replace {words} by {c}'.format(words=top_words, c = concept))
            #print('--')
    new_df['replaced'] = new_words
    replacement = pd.DataFrame({'to_replace': to_replace, 'replaced_by':replaced_by})
    return replacement, new_df

"Extract all nouns and verbs from the dataset"

def get_wordnet_onto(treebank_tag):
    """Map ['NN', 'NNS', 'NNP', 'NNPS'] to NOUN....."""
    if treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    else:
        return None

def processing_onto(sentence, tokenizer, stop, lemmatiser):
    
    "Tokenization"
    sentence = tokenizer.tokenize(sentence)
    
    "Lemmatization"
    tokens_pos = pos_tag(sentence)
    tokens_pos = [(w,get_wordnet_onto(p)) for (w,p) in tokens_pos]
    sentence = [lemmatiser.lemmatize(w, pos=p) for (w,p) in tokens_pos if p != None]
    
    "Stopwords removing"
    sentence = [x for x in sentence if x not in stop ]
    
    return ' '.join(sentence)

def extracting_onto(df, target_col, result_col):
    tokenizer = RegexpTokenizer(r'\w+')
    stop = stopwords.words('english')
    lemmatiser = WordNetLemmatizer()
    
    new_text = []
    for responses in df[target_col]:
        processed_responses = []
        for response in responses:
            precessed_res = ''
            for res in response.split('.'):
                precessed_res += processing_onto(res, tokenizer, stop, lemmatiser) + ' '
            processed_responses.append(precessed_res)
        new_text.append(processed_responses)
    df[result_col] = new_text
    return df



def count(df, textfield, replacement=False, replace=None):
    """
        Count how many nouns and verbs present in the dataset
        If replacement is True, we first replace nouns and verbs based 
        on `replace dictionary` then count
    
    """
    new_df = df.copy()
    new_df['relevant'] = new_df[textfield].apply(lambda x : ' '.join(x))
    tf_vectorizer = CountVectorizer()
    
    if replacement: # we do replacement
        result = []
        for resp in new_df['relevant']:
            for key in replace.keys():
                resp = resp.replace(key, replace[key])
            result.append(resp)
        new_df['ontology_replacement'] = result
        
        tf = tf_vectorizer.fit_transform(new_df['ontology_replacement'])
    else :
        tf = tf_vectorizer.fit_transform(new_df['relevant'])
    
    
    tf_feature_names = tf_vectorizer.get_feature_names()
    
    print('There are {c} nouns and verbs in the dataset'.format(c = tf.shape[1]))
    return tf_feature_names


def find_top_most_similarity(tf_feature_names, df, threshold):
    replace = {}
    features = np.array(tf_feature_names)
    X = [x.tolist() for x in df['fasttext']]
    distances_matrix = cosine_similarity(X, X)
    for idx, noun in enumerate(features):
        sim = distances_matrix[idx,:]
        min_distances = np.where(sim > threshold)[0]
        
        top_words = features[min_distances]
        top_scores= sim[min_distances]

        if (len(top_words) > 1):
            replace_ind = top_scores.argsort()[::-1][1]
            print('=====> We can replace `{n1}` by `{n2}`'.format(n1=noun,n2=top_words[replace_ind]))
            print('---------------------')
            if noun in replace or top_words[replace_ind] in replace:
                continue
            else:
                replace[noun] = top_words[replace_ind]
                
    return replace