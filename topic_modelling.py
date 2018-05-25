# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from helpers import *
import numpy as np
import pandas as pd 

def LDA(df, week, questionNb, nbTopic, n, plt, replacement=False, replace=None):
    """
        df: dataframe contains documents/sentences
        week: which week that you want ?
        questionNb: which question in this week ?
        nbTopic: how many topics do you think these document have ?
        n: find top n sentences contribute for each topic
    """
    
    print('Welcome to LDA algorithm.')
    print('Begin find topics for all answers of question number {i} of week {w}'.format(i=questionNb, w=week))
    
    df = df[df['week']==week]
    df['relevant'] = df['processed_responses'].apply(lambda x : x[questionNb-1] if len(x) > questionNb-1 else '')    
    df = df.reset_index().drop('index', 1)

    "Do the noun replacement if needed"
    if replacement:
        replaced = []
        for resp in df['relevant']:
            for key in replace.keys():
                resp = resp.replace(key, replace[key])
            replaced.append(resp)
        df['ontology_replacement'] = replaced
        tf_vectorizer = CountVectorizer()
        tf = tf_vectorizer.fit_transform(df['ontology_replacement'])
        tf_feature_names = tf_vectorizer.get_feature_names()
    
    else:
	    " LDA can only use raw term counts for LDA because it is a probabilistic graphical model"
	    tf_vectorizer = CountVectorizer()#(min_df = 7, max_df = 18)
	    tf = tf_vectorizer.fit_transform(df['relevant'])
	    tf_feature_names = tf_vectorizer.get_feature_names()
    
    print('Shape of tfidf matrix:', tf.shape)
    
    "LDA"
    lda = LatentDirichletAllocation(n_topics=nbTopic, max_iter=50, learning_method='online', learning_offset=50.,random_state=nbTopic*50).fit(tf)

    print('Topic words distribution shape:', lda.components_.shape)
    plot_topics(lda, tf_feature_names, nbTopic, plt)
    
    sim_matrix = concepts_responses_similarity(lda.components_, tf.toarray())
    
    "Retrieve top n responses for a specific topic"
    d = {}
    for i in range(nbTopic):
        response_scores = sim_matrix[:,i]
        top_indexes = response_scores.argsort()[-n:][::-1]
        top_responses = []
        for index in top_indexes:
            top_responses.append(df.iloc[index]['standardized_responses'])
        d['Topic #'+str(i)] = top_responses
    
    return df, pd.DataFrame.from_dict(d), lda.transform(tf)


def nmf(df, week, questionNb, nbTopic, n, plt):
    """
        df: dataframe contains documents/sentences
        week: which week that you want ?
        questionNb: which question in this week ?
        nbTopic: how many topics do you think these document have ?
        n: find top n sentences contribute for each topic
    """
    
    
    print('Welcome to NMF algorithm.')
    print('Begin find topics for all answers of question number {i} of week {w}'.format(i=questionNb, w=week))
    
    df = df[df['week']==week]
    
    df['relevant'] = df['processed_responses'].apply(lambda x : x[questionNb-1] if len(x) > questionNb-1 else '')
    
    df = df.reset_index().drop('index', 1)
    
    " Non-negative Matrix Factorization is able to use tf-idf "
    
    tfidf_vectorizer = TfidfVectorizer(min_df = 7, max_df = 18)#(max_features=vocab_size)
    tfidf = tfidf_vectorizer.fit_transform(df['relevant'])
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    
    print('Shape of tfidf matrix:', tfidf.shape)
    
    "NMF"
    nmf = NMF(n_components=nbTopic, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
    
    print('Topic words distribution shape:', nmf.components_.shape)
    plot_topics(nmf, tfidf_feature_names, nbTopic, plt)
    
    sim_matrix = concepts_responses_similarity(nmf.components_, tfidf.toarray())
    
    "Retrieve top n responses for a specific topic"
    d = {}
    for i in range(nbTopic):
        response_scores = sim_matrix[:,i]
        top_indexes = response_scores.argsort()[-n:][::-1]
        top_responses = []
        for index in top_indexes:
            top_responses.append(df.iloc[index]['standardized_responses'])
        d['Topic #'+str(i)] = top_responses
    
    
    return df, pd.DataFrame.from_dict(d), nmf.transform(tfidf)

def plot_topics(model, vocab, nbTopics, plt, n_terms=10, nbColsPlot=4):
    """
        model: LDA or NMF model
        vocab: features (words)
        nbTopics: number of topics
        n_terms: n terms for each topic
        nbColsPlot: number of columns 
        
    """

    topics = model.components_
    for row in range(0, nbTopics, nbColsPlot):
        plot_topic(row, vocab, topics, nbTopics, n_terms, nbColsPlot, plt)



def plot_topic(row, vocabulary, topics, nbTopics, n_terms, nbColsPlot, plt):
    MAGIC_NUMBER = 50
    fontsize_init = MAGIC_NUMBER
    fig = plt.figure(figsize=(15, 7))
    for topic in range(row, min(row + nbColsPlot, nbTopics)):

        plt.subplot(1, nbColsPlot, topic%nbColsPlot + 1)
        plt.ylim(0, n_terms + .5)
        plt.xticks([]) 
        plt.yticks([])
        plt.title('Topic #{}'.format(topic), fontsize=30)
        top_index = topics[topic].argsort()[:-n_terms - 1:-1]
        scores = topics[topic][top_index]
        words_topic = []
        max_topic_score = max(scores)
        for ind in range(len(top_index)):
            font_size = fontsize_init*scores[ind]/max_topic_score
            font_size = min(font_size, MAGIC_NUMBER)
            font_size = max(font_size, 20)
            words_topic.append(vocabulary[top_index[ind]])
            plt.text(0.05, n_terms-ind-0.5, vocabulary[top_index[ind]], fontsize=font_size) 

    plt.tight_layout()
    plt.show()



def similarity(vec1, vec2):
    """
        Similarity between 2 vector
    """
    result = np.dot(vec1, vec2) / np.sqrt(np.sum(vec1 ** 2) * np.sum(vec2 ** 2))
    if (np.isnan(result)) : 
        result = 0
    return result

def concepts_responses_similarity(lda, tfidf):
    """
        Return a similarity matrix:
            - Row : reponses
            - Col : topics
    """
    nbResponses, nbFeature = tfidf.shape
    nbTopics, nbFeature = lda.shape
    result = np.zeros((nbResponses,nbTopics))
    for i in range(nbResponses):
        for j in range(nbTopics):
            result[i,j] = similarity(tfidf[i],lda[j])
    
    return result


"""
    Top nbTerm-responses with the highest similarity with topic topic_number
"""
def print_top_responses_for_topic(df, topic_number, nbTerm):
    print('Top', nbTerm ,"responses that contribute for Topic #{v}".format(v=topic_number))
    print('***************')
    print('***************')
    print('***************')
    column = 'Topic #' + str(topic_number)
    for i in range(nbTerm):
        print(df.iloc[i][column])
        print('--------------------------------------------------')

"""
    Find which topics have the most contribution for a specific response
"""
def print_top_topics_for_response(topicsSent, responseID, n):
    show = "Response "+str(responseID)+": "
    sentence = topicsSent[responseID,:]
    top_indexes = sentence.argsort()[::-1][:n]
    for idx, i in enumerate(top_indexes):
        show += str(sentence[i])[:4] + '*Topic' + str(i)
        if (idx != n - 1):
            show += ' + '
    print(show)
