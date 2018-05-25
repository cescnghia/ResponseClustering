# -*- coding: utf-8 -*-

import nltk
from nltk import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import wordnet, stopwords
from tags_processing import get_wordnet_pos
from helpers import flatten
import re


def standardize_responses(df, text_field):
    
    urls_re = r'http\S+' #urls
    ht_re = r'#\S+'      #hashtags
    at_re = r'@\S+'      #@
    nb_re = " \d+"       #numbers
    
    punctuations = ["/","(",")","\\","|", ":",",",";","?", "!", "[", "]", "{","}"]
    
    new_text = []
    for text in df[text_field]:
        standardized = []
        for te in text:
            #text = '. '.join(text)
            te = te.replace("\\n"," ").replace("\\r"," ").replace("\n", " ").replace("-"," ").replace("_"," ")

            for x in punctuations:
                te = te.replace(x , " ")

            te = re.sub(urls_re , " ", te)
            te = re.sub(ht_re   , " ", te)
            te = re.sub(at_re   , " ", te)
            te = re.sub(nb_re   , " ", te)

            te = te.lower()
            standardized.append(te)
        
        new_text.append(standardized)
    
    df['standardized_responses'] = new_text
    return df


def processing_response(sentences, tokenizer, lemmatiser, stop):
    
    processed = []
    for sentence in sentences.split('.'):
    
        "Tokenization"
        sentence = tokenizer.tokenize(sentence)

        "Lemmatization"
        tokens_pos = pos_tag(sentence)
        tokens_pos = [(w,get_wordnet_pos(p)) for (w,p) in tokens_pos]
        sentence = [lemmatiser.lemmatize(w, pos=p) for (w,p) in tokens_pos if p != None]

        "Stopwords removing"
        sentence = [x for x in sentence if x not in stop]
        
        processed.append(sentence)
      
    return ' '.join(flatten(processed))



def processing_responses(df, text_field):
    new_text = []
    
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatiser = WordNetLemmatizer()
    stop = stopwords.words('english')
    
    for text in df[text_field]:
        processed = []
        # procese one answer at a time
        # an answer can have multiple sentences
        for answer in text: 
            processed.append(processing_response(answer, tokenizer, lemmatiser, stop))
        new_text.append(processed)

    df['processed_responses'] = new_text
    return df