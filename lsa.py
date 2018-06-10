from scipy.sparse.linalg import svds
import operator
import nltk
from nltk import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import wordnet, stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import *
from french_preprocessing import *

import numpy as np



def response_term_similarity(u, S, v_t):
    """
      Similarity between a response and term in concept space.
      (We apply SVD algorithm to original matrix X.)
      Here `u`   is a representation of a response in concept space
           `v_t` is a representation of a term in concept space
            S     is sigular values of X
    """
    temp = np.diag(S) @ v_t
    return np.dot(u, temp) / np.sqrt(np.sum(u ** 2) * np.sum(temp **2))


def response_response_similarity(resp1, resp2, S):
    resp1_ = np.diag(S) @ resp1
    resp2_ = np.diag(S) @ resp2
    
    return np.dot(resp1_, resp2_) / np.sqrt(np.sum(resp1_ ** 2) * np.sum(resp2_ ** 2))


def check_responses(df, text_field, latentSpace, yourConcept, POS_JAR=None, POS_MODEL=None, english=True):

    ##############PREPROCESSING TEXT##############
    "Create a dataframe from yourConcept and concatenate with df"
    NB_REPLICATE = 10
    df_concept = pd.DataFrame([yourConcept] * NB_REPLICATE, columns=[text_field])
    df = pd.concat([df, df_concept]).reset_index().drop('index', 1)
    if (english):
        df = standardize_text(df, text_field)
        df, _ = processing_text(df, 'standardized_text')
    else:
        df = french_standardize_text(df, text_field)
        df = french_processing_text(df, 'standardized_text', POS_JAR, POS_MODEL)
    
    "Students' response scores"
    response_scores = {}
    
    "Compute TFIDF matrix"
    tfidf_vec = TfidfVectorizer()
    tfidf = tfidf_vec.fit_transform(df['processed_text'])
    vocab = tfidf_vec.get_feature_names()
    
    #tfidf = tfidf[:-NB_REPLICATE]
    
    
    
    "apply SVD to TFIDF matrix"
    U, S, V_T = svds(tfidf, k=latentSpace)
    U = U[:-NB_REPLICATE,:]

    nbResponse, _ = U.shape
    _, nbTerm     = V_T.shape
    print('There are {r} responses in {t} significant terms'.format(r=nbResponse,t=nbTerm))

    df = df[:-NB_REPLICATE].reset_index().drop('index', 1)
    
    "Preprocessed yourConcept"
    tokenizer = RegexpTokenizer(r'\w+')
    if (english):
        lemmatiser = WordNetLemmatizer()
        stop = stopwords.words('english')
        _, concepts = english_processing_sentence(yourConcept, tokenizer, lemmatiser, stop)
    else:
        pos_tagger = StanfordPOSTagger(POS_MODEL, POS_JAR, encoding='utf8' )
        lemmatiser = FrenchLefffLemmatizer()
        stop = stopwords.words('french')
        concepts = french_processing_sentence(yourConcept, tokenizer, pos_tagger, lemmatiser, stop)
    scores = []
    "Compute score for each responses"
    for response in range(nbResponse):
        score = 0.0
        for concept in concepts:
            try:
                concept_idx = vocab.index(concept)
            except:
                continue
                
            u = U[response]
            v = V_T.T[concept_idx]
            score += response_term_similarity(u, S, v)
        
        scores.append(score)
        
    df['score'] = scores
    
    
    "Compute students' similarity score"
    
    sim_scores = np.zeros((nbResponse,nbResponse))
    
    for row in range(nbResponse):
        for col in range(row + 1):
            resp1 = U[row]
            resp2 = U[col]
            score = response_response_similarity(resp1, resp2, S)
            sim_scores[row, col] = score
            sim_scores[col, row] = score
    
    
    return df, sim_scores