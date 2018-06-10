from sklearn.decomposition import PCA
import pandas as pd
from preprocessing import *
from helpers import *
from concept import *
from ontology import *



def get_dashboard(texts, 
                  textfield, 
                  fasttext, 
                  stop_words, 
                  k, 
                  threshold_replace, 
                  seed, 
                  cutoff=False, 
                  threshold_takeout=None):
    """
        -Input:
            texts: array of string
            textfield: text column's name (for creating a dataframe)
            k: number of cluster
            threshold_replace: the threshold (similarity) in which 
                               we decide to replace words
            threshold_takeout: frequency to take out common words (iff cutoff is True)
        -Output:
            processed Text, dashboard and replacement dataframes
    """

    
    
    ##############PREPROCESSING TEXT##############
    print('Begin preprocessing stage')
    new_df = pd.DataFrame(texts, columns=[textfield])
    new_df = standardize_text(new_df, textfield)
    new_df, normalised_df = processing_text(new_df, 'standardized_text')
    print(len(normalised_df))
    
    ##############GLOBAL STOPWORDS REMOVING##############
    normalised_df['isFrequency'] = normalised_df['processed_text'].apply(lambda x : x in stop_words)
    normalised_df = normalised_df[normalised_df['isFrequency']==False]
    normalised_df = normalised_df.drop('isFrequency', axis=1)
    print(len(normalised_df))
    ###############WORDS EMBEDDING WITH FASTTEXT##############
    print('Begin embedding stage')
    vects = []
    presents = []
    for i, row in normalised_df.iterrows():
        word = row['processed_text']
        if word in fasttext:
            vects.append(fasttext[word])
            presents.append(1)
        else:
            presents.append(0)
            vects.append(0)
    
    normalised_df['vect'] = vects
    normalised_df['present'] = presents
    normalised_df = normalised_df[normalised_df['present']==1]
    normalised_df = normalised_df.drop('present', axis=1)
    print(len(normalised_df))
    ###############WORDS REPLACEMENT##############
    print('Begin words replacement stage')
    print('***  Before the replacement, we have {w} words'.format(w=len(normalised_df)))
    replacement_df, normalised_df = find_replacement(normalised_df, threshold=threshold_replace)

    normalised_df = normalised_df.groupby(['replaced']).agg({'count':'sum','pos': lambda x : ' '.join(x)})
    normalised_df = normalised_df.reset_index()
    normalised_df.columns = ['processed_text', 'count', 'pos']
    normalised_df['pos'] = normalised_df['pos'].apply(lambda x : x.split()[0])
    print(len(normalised_df))
    "Re-embedding for new words (after replacing word)"
    vects = []
    presents = []
    for i, row in normalised_df.iterrows():
        word = row['processed_text']
        if word in fasttext:
            vects.append(fasttext[word])
            presents.append(1)
        else:
            presents.append(0)
            vects.append(0)
    
    normalised_df['vect'] = vects
    normalised_df['present'] = presents
    normalised_df = normalised_df[normalised_df['present']==1]
    normalised_df = normalised_df.drop('present', axis=1)
    print('***  After the replacement, we have {w} words'.format(w=len(normalised_df)))
    
    ###############REMOVE TOP FREQUENCY WITHIN THE DATASET##############
    print('Begin remove most frequency stage (within the dataset)')
    if cutoff:
        normalised_df = normalised_df[normalised_df['count'] < threshold_takeout]
        normalised_df = normalised_df.reset_index().drop('index', 1)
    
    ###############CLUSTERING##############
    print('Begin clustering stage')
    
    X = []
    for vect in normalised_df['vect']:
        X.append(list(vect))
    
    "PCA on X"
    pca = PCA(n_components=2)
    pca_x = pca.fit_transform(X)
    
    "Cluster"
    labels, centers, clusters = k_means(K=k, data=pca_x, seed=seed)

    
    
    "Store the result of k-means algorithm"
    normalised_df['x'] = pca_x[:,0]
    normalised_df['y'] = pca_x[:,1]
    normalised_df['cluster'] = labels
    normalised_df['cluster_center'] = normalised_df['cluster'].apply(lambda x : centers[x])

    
    ##############VISUALIZATION##############
    print('Begin visualization stage')
    normalised_df = find_concept_wordnet(k, normalised_df, max_depth=3)

    
    print('Finish')
    print('--------------------------------------')
    
    #####RELEVANT COLUMNS###
    #normalised_df = normalised_df[['processed_text', 'count','pos', 'x', 'y', 'cluster', 'cluster_center', 'wordnet']]
    
    return new_df, replacement_df, normalised_df



def update_dashboard(texts, 
                     textfield,
                     old_df,
                     old_replacement_df,
                     old_normalised_df,
                     fasttext, 
                     stop_words, 
                     k, 
                     threshold_replace, 
                     seed, 
                     cutoff=False, 
                     threshold_takeout=None):
    """
        texts: new data (array of string)
        textfield: columns name (for creating dataframe)
    """
    
    
    ##############UPDATE##############
    
    "Preprocessing new dataset"
    print('Begin preprocessing stage')
    new_df = pd.DataFrame(texts, columns=[textfield])
    new_df = standardize_text(new_df, textfield)
    new_df, new_normalised_df = processing_text(new_df, 'standardized_text')
    
    "Global Stopwords Removing"
    print('Begin global stopwords removing stage')

    new_normalised_df['isFrequency'] = new_normalised_df['processed_text'].apply(lambda x : x in stop_words)
    new_normalised_df = new_normalised_df[new_normalised_df['isFrequency']==False]
    new_normalised_df = new_normalised_df.drop('isFrequency', axis=1)
    
    "Concatenate old and new data then groupby"
    print("Begin concatenate old and new data then groupby stage")
    new_df = pd.concat([old_df, new_df]).reset_index().drop('index', 1)
    old_normalised_df = old_normalised_df[['processed_text', 'count', 'pos']]
    
    new_normalised_df = pd.concat([old_normalised_df, new_normalised_df]).reset_index().drop('index', 1)
    a = new_normalised_df.groupby(['processed_text'])['count'].agg(sum)
    a = a.reset_index()
    b = new_normalised_df.drop_duplicates('processed_text')[['processed_text','pos']]
    new_normalised_df = pd.merge(a, b, on='processed_text', how='outer')

    "Word embedding with Fasttext"
    print('Begin embedding stage')
    vects = []
    presents = []
    for i, row in new_normalised_df.iterrows():
        word = row['processed_text']
        if word in fasttext:
            vects.append(fasttext[word])
            presents.append(1)
        else:
            presents.append(0)
            vects.append(0)
    
    new_normalised_df['vect'] = vects
    new_normalised_df['present'] = presents
    new_normalised_df = new_normalised_df[new_normalised_df['present']==1]
    new_normalised_df = new_normalised_df.drop('present', axis=1)
    #print(new_normalised_df.head())
    "Words replacement"
    print('Begin words replacement stage')
    print('***  Before the replacement, we have {w} words'.format(w=len(new_normalised_df)))
    new_replacement_df, new_normalised_df = find_replacement(new_normalised_df, threshold=threshold_replace)
    #print(len(replacement_df))
    new_replacement_df = pd.concat([old_replacement_df, new_replacement_df]).reset_index().drop('index', 1)
    new_normalised_df = new_normalised_df.groupby(['replaced']).agg({'count':'sum','pos': lambda x : ' '.join(x)})
    new_normalised_df = new_normalised_df.reset_index()#.drop('index', 1)
    new_normalised_df.columns = ['processed_text', 'count', 'pos']
    new_normalised_df['pos'] = new_normalised_df['pos'].apply(lambda x : x.split()[0])
    "Re-embedding for new words (after replacing word)"
    vects = []
    presents = []
    for i, row in new_normalised_df.iterrows():
        word = row['processed_text']
        if word in fasttext:
            vects.append(fasttext[word])
            presents.append(1)
        else:
            presents.append(0)
            vects.append(0)
    
    new_normalised_df['vect'] = vects
    new_normalised_df['present'] = presents
    new_normalised_df = new_normalised_df[new_normalised_df['present']==1]
    new_normalised_df = new_normalised_df.drop('present', axis=1)
    
    ###############REMOVE TOP FREQUENCY WITHIN THE DATASET##############
    print('Begin remove most frequency stage (within the dataset)')
    if cutoff:
        new_normalised_df = new_normalised_df[new_normalised_df['count'] < threshold_takeout]
        new_normalised_df = new_normalised_df.reset_index().drop('index', 1)
    
    "Clustering"
    print('Begin clustering stage')
    X = []
    for vect in new_normalised_df['vect']:
        X.append(list(vect))

    
    "PCA on X"
    pca = PCA(n_components=2)
    pca_x = pca.fit_transform(X)
    
    "Cluster"
    labels, centers, clusters = k_means(K=k, data=pca_x, seed=seed)

    
    
    "Store the result of k-means algorithm"
    new_normalised_df['x'] = pca_x[:,0]
    new_normalised_df['y'] = pca_x[:,1]
    new_normalised_df['cluster'] = labels
    new_normalised_df['cluster_center'] = new_normalised_df['cluster'].apply(lambda x : centers[x])
    
    "Find concept"
    print('Begin find concept stage')
    #cluster_colors = generating_k_colors(k)
    new_normalised_df = find_concept_wordnet(k, new_normalised_df, max_depth=3)
    
    #####RELEVANT COLUMNS###
    #new_normalised_df = new_normalised_df[['processed_text', 'count','pos', 'x', 'y', 'cluster', 'cluster_center', 'wordnet']]
    
    return new_df, new_replacement_df, new_normalised_df

def zoom_cluster(normalised_df, number):
    
    """
        PATH_IN: path contain dashboard data (.csv file)
        numbre : the cluster number want to zoom in
        PATH_OUT: path to write the result
    """
    
    

    new_df = normalised_df[normalised_df['cluster']==number]
    new_df = new_df.reset_index().drop('index', 1)

    x = []
    for vect in new_df['vect']:
        #vect = vect[1:-1].split()        
        x.append(list(vect))
    
    pca = PCA(n_components=2)
    pca_fitted = pca.fit_transform(x)
    #tsne_model = TSNE(perplexity=dataframe['cluster'].max()+1, n_components=2, init='pca', method='exact',n_iter=2000)
    #tsne_fitted = tsne_model.fit_transform(x)

    new_df['x_cluster'] = pca_fitted[:,0] #tsne_fitted[:,0]
    new_df['y_cluster'] = pca_fitted[:,1] #tsne_fitted[:,1]
    
    columns = ['processed_text', 'count', 'x_cluster', 'y_cluster', 'wordnet']
    
    return new_df[columns]
    