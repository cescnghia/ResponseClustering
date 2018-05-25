# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from textblob import TextBlob



def plot_distribution(plt, serie, title):
    """
        serie: padans serie to plot
        title: title of the plot
    """
    plt.subplots(figsize=(15,8))
    serie.hist(bins=serie.max() ,edgecolor='black')
    plt.xticks(list(range(0,serie.max())))
    plt.title(title)
    plt.show()

def plot_frequency(serie, title, plt, n_terms = 15):
    """
        serie: pandas serie to plot
        title: title of the plot
        n_terms: plot top n-terms frequency
    """
    terms = []

    for text in serie:
        for term in text.split():
            terms.append(term)

    plt1 = pd.Series(terms).value_counts().sort_values(ascending=False)[:n_terms].to_frame()
    sns.barplot(plt1[0],plt1.index,palette=sns.color_palette('inferno_r',n_terms))
    plt.title(title)
    fig=plt.gcf()
    fig.set_size_inches(20,n_terms-5)
    plt.show()

def plot_frequency_week(df, week, questionNb, plt, n_terms = 15):
    df = df[df['week']==week]
    terms = []

    for text in df['processed_responses']:
        if len(text) > questionNb - 1:
            text = text[questionNb - 1]
            for term in text.split():
                terms.append(term)

    plt1 = pd.Series(terms).value_counts().sort_values(ascending=False)[:n_terms].to_frame()
    sns.barplot(plt1[0],plt1.index,palette=sns.color_palette('inferno_r',n_terms))
    plt.title('Top terms frequency of week ' + str(week))
    fig=plt.gcf()
    fig.set_size_inches(20,n_terms-5)
    plt.show()



def k_means(K, data, seed):
    """
        Input:
            - K : number of cluster
            - data : List of words' vectors
            - seed : set random seed 
            
        Output:
            - labels : cluster assignment for words
            - centers: center of K cluster
            - clusters: a dictionary key = cluster, value = list of words within cluster
    """

    kmeans = KMeans(n_clusters=K, random_state=seed).fit(data)
    labels = kmeans.labels_
    centers= kmeans.cluster_centers_
    

    clusters = {}

    for index, label in enumerate(labels):
        if label in clusters:
            temp = clusters[label]
            temp.append(index)
            clusters[label] = temp
        else:
            clusters[label] = [index]
   
    return labels, centers, clusters



def print_cluster(K, clusters, vocabulary, truncated=False, n=None):
    """
        K: number of cluster
        clusters: output of k_means function
        vocabulary: numpy array contains words
        truncated: if we want to truncate the print with first n terms
    """
    
    for key in range(K):
        string = ''
        indexes = clusters[key]
        if truncated:
            indexes = indexes[:n]
        print('Cluster number: ',key)
        
        for index in indexes:
            string += vocabulary[index]+', '
        print('[', string, ']')
        print('\n')

def detect_language(x):
    if len(x) == 0:
        return 'vi'
    else:
        return TextBlob(x[0]).detect_language()


def flatten(array):
    return [s for sublist in array for s in sublist]


def concatenate_responses(responses):
    """
        Concatenate responses into a list. Excluded QMC's answers
        list[0] = answer for the first question
        list[1] = answer for the second question...
    """
    try:
        responses = list(json.loads(responses.replace('\\\\"','')).values())
    except :
        responses =  ''
        
    return [ x for x in responses if len(x) > 10]


def read_data_responses(PATH):
    """First, open file inqreflection.tsv, 
        delete some irrelevant row from line 2400 to 2409"""

    ref = pd.read_csv(PATH, header = None, sep = '\t', encoding = "ISO-8859-1")
    ref = ref.iloc[1:].reset_index().drop('index', 1)
    ref.columns = ['week', 'text']
    ref['week'] = ref['week'].astype(dtype=int)
    ref['text'] = ref['text'].apply(concatenate_responses)
    
    ref['language'] = ref['text'].apply(detect_language)
    ref = ref[ref['language']=='en']
    ref = ref.reset_index().drop('index', 1)
    
    return ref

def select_sub_responses(df, week, num):
    new_df = df.copy()
    new_df = new_df[new_df['week'] == week]
    new_df['text'] = new_df['text'].apply(lambda x : x[num-1])
    new_df = new_df.reset_index().drop('index', 1)
    return new_df



def generating_k_colors(K):
    "Generate k random colors"
    r = lambda: np.random.randint(0,255)

    colors = ['#%02X%02X%02X' % (r(),r(),r()) for x in range(K)]
    
    return colors