# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd 
from nltk.corpus import wordnet as wn
from nltk import pos_tag
import requests

def distance_to_cluster_center(row):
    "Compute Euclidean distance"
    return np.sum(([row['x'], row['y']] - row['cluster_center']) ** 2)



def find_representative(K, df):
    """
        K : number of cluster
        df: dataframe
    """

    has_representative = K*[False]
    min_dist = df.groupby('cluster')['distance_to_center'].min()
    result = []
    
    for ind, row in df.iterrows():
        cluster = row['cluster']
        if (has_representative[cluster]):
            result.append(False)
        else:
            has_representative[cluster] = (min_dist[cluster] == row['distance_to_center'])
            result.append(has_representative[cluster])

    df['representative_cluster'] = result
    return df




def remove_articles(sentence):
    """
        Detect if an indefinite article presents and transform `a programming language` to `programming language`
    """
    tokens = sentence.split()
    if tokens[0] == 'a':
        return sentence[2:]
    elif tokens[0] == 'an':
        return sentence[3:]
    else:
        return sentence


def related_words(word, n):
    """
        word: word to be passed to conceptnet.io
        n: take n top closest words to word
    """
    result = []
    obj = requests.get('http://api.conceptnet.io/c/en/' + word).json()['edges']
    n = min(n, len(obj))
    obj = obj[:n]
    for i in range(n):
        response = obj[i]['surfaceText']

        if response: #and ('is a type of' in response or 'is related to' in response) :
            if response.index('[') == 0 :
                tokens = response.split(']]')
                first = tokens[0].split('[[')[1]
                last  = tokens[1].split('[[')[1]
                first = remove_articles(first).lower()
                last  = remove_articles(last).lower()
                if word in first :
                    result.append(last)
                elif word in last :
                    result.append(first)
    return list(set(result))


def find_concept_conceptnet(K, df):
    """
        K : number of cluster
        df: dataframe contains column fasttext vector of words
    """

    "Distance of this words to the cluster center"
    df['distance_to_center'] = df.apply(distance_to_cluster_center, axis=1)


    "Find the center word for each cluster"
    df = find_representative(K, df)


    df['related_words' ] = df['processed_text'].apply(lambda x : related_words(x, 20))


    central_words = {}

    for i in range(K):
        central_words[i] = []
        cluster_df = df[df['cluster']==i]
        for ind, word in cluster_df.iterrows():
            temp = central_words[i]
            temp.append(word['related_words'])
            central_words[i] = temp
        
    for i in range(K):
        tokens = central_words[i]
        tokens = [c for b in tokens for c in b]
        central_words[i] = max(set(tokens), key=tokens.count)

    df['conceptnet'] = df['cluster'].apply(lambda x : central_words[x])

    return df



def find_concept_wordnet(K, df, max_depth):
    """
        K : number of cluster
        df: dataframe contains column fasttext vector of words
    """

    "Distance of this words to the cluster center"
    df['distance_to_center'] = df.apply(distance_to_cluster_center, axis=1)


    "Find the center word for each cluster"
    df = find_representative(K, df)

    df['wordnet'] = ''
    for i in range(K):
        new_df = df[df['cluster']==i]
        synset_words = []
        for i, row in new_df.iterrows():
            pos = row['pos']
            try:
                if pos == 'v':
                    syn = wn.synset(row['processed_text'] + '.v.01')
                else:
                    syn = wn.synset(row['processed_text'] + '.n.01')
                synset_words.append(syn)
            except:
                pass

        dict_array = []
        length = len(synset_words)
        for i, v in enumerate(synset_words):
            for j in range(i+1, length):
                dict_array.append(v._shortest_hypernym_paths(synset_words[j]))
            
        all_hypernyms = []
        for d in dict_array:
            for (k,v) in d.items():
                if (v < max_depth):
                    all_hypernyms.append(k.name().split('.')[0])
                    
        try:
            concept =  max(set(all_hypernyms), key=all_hypernyms.count)
            if concept == '*ROOT*':
                while concept in all_hypernyms:
                    all_hypernyms.remove(concept)
                concept = max(set(all_hypernyms), key=all_hypernyms.count)
        except:
            concept = 'Not Found'
        
        df['wordnet'][new_df.index] = concept

    return df


def find_hypernym(words, pos, max_depth):
    """
        Find hypernym from a list of words and its POS
    """

    synset_words = []
    for i, word in enumerate(words):
        p = pos[i]
        try:
            if p == 'v':
                syn = wn.synset(word + '.v.01')
            else:
                syn = wn.synset(word + '.n.01')
            synset_words.append(syn)
        except:
            pass

    dict_array = []
    length = len(synset_words)
    for i, v in enumerate(synset_words):
        for j in range(i+1, length):
            dict_array.append(v._shortest_hypernym_paths(synset_words[j]))
            
    all_hypernyms = []
    for d in dict_array:
        for (k,v) in d.items():
            if (v < max_depth):
                all_hypernyms.append(k.name().split('.')[0])
                    
    try:
        concept =  max(set(all_hypernyms), key=all_hypernyms.count)
        if concept == '*ROOT*':
            while concept in all_hypernyms:
                all_hypernyms.remove(concept)
            concept = max(set(all_hypernyms), key=all_hypernyms.count)
    except:
        concept = 'Not Found'
    
    return concept    