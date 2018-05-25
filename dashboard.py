from sklearn.decomposition import PCA
import pandas as pd
from preprocessing import *
from helpers import *
from concept import *
from ontology import *



def get_dashboard(PATHS, textfield, fasttext, k, threshold_replace, threshold_takeout, seed):
    """
        -Input:
            PATHS: list of 6 paths (.csv files)
                + path 0: path to data
                + path 1: path to english most frequency words
                + path 2: path for storing processed text data
                + path 3: path for storing word replacement
                + path 4: path of dashboard a (removed top frequency with threshold `threshold_takeout`)
                + path 5: path of dashboard b (dont remove top frequency )
            textfield: text column's name (column consists rows of string)
            k: number of cluster
            threshold_replace: the threshold (similarity) in which 
                               we decide to replace words
            threshold_takeout: frequency to take out common words
        -Output:
            4 .csv files
    """
    new_df = pd.read_csv(PATHS[0])
    
    ##############PREPROCESSING TEXT##############
    print('Begin preprocessing stage')
    new_df = standardize_text(new_df, textfield)
    new_df, normalised_df = processing_text(new_df, 'standardized_text')
    
    ##############GLOBAL STOPWORDS REMOVING##############
    english_most_frequency = pd.read_csv(PATHS[1],sep = ',', encoding = "ISO-8859-1")
    stop_words = [x for x in english_most_frequency['Word']]
    normalised_df['isFrequency'] = normalised_df['processed_text'].apply(lambda x : x in stop_words)
    #print(normalised_df[normalised_df['isFrequency']==True]['processed_text'])
    normalised_df = normalised_df[normalised_df['isFrequency']==False]
    normalised_df = normalised_df.drop('isFrequency', axis=1)
    
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
    
    ###############WORDS REPLACEMENT##############
    print('Begin words replacement stage')
    print('***  Before the replacement, we have {w} words'.format(w=len(normalised_df)))
    replacement_df, normalised_df = find_replacement(normalised_df, threshold=threshold_replace)

    normalised_df = normalised_df.groupby(['replaced']).agg({'count':'sum','pos': lambda x : ' '.join(x)})
    normalised_df = normalised_df.reset_index()
    normalised_df.columns = ['processed_text', 'count', 'pos']
    normalised_df['pos'] = normalised_df['pos'].apply(lambda x : x.split()[0])
    
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
    print('Begin remove most frequency stage')
    normalised_df_a = normalised_df[normalised_df['count'] < threshold_takeout]
    normalised_df_b = normalised_df.copy()
    print('***  We removed {w} most frequency words'.format(w=len(normalised_df_b)-len(normalised_df_a)))
    
    
    ###############CLUSTERING##############
    print('Begin clustering stage')
    X_a = []
    for vect in normalised_df_a['vect']:
        X_a.append(list(vect))
    
    X_b = []
    for vect in normalised_df_b['vect']:
        X_b.append(list(vect))
    
    "PCA on X"
    pca = PCA(n_components=2)
    pca_a = pca.fit_transform(X_a)
    pca_b = pca.fit_transform(X_b)
    
    "Cluster"
    labels_a, centers_a, clusters_a = k_means(K=k, data=pca_a, seed=seed)
    labels_b, centers_b, clusters_b = k_means(K=k, data=pca_b, seed=seed)
    
    
    "Store the result of k-means algorithm"
    normalised_df_a['x'] = pca_a[:,0]
    normalised_df_a['y'] = pca_a[:,1]
    normalised_df_a['cluster'] = labels_a
    normalised_df_a['cluster_center'] = normalised_df_a['cluster'].apply(lambda x : centers_a[x])
    normalised_df_b['x'] = pca_b[:,0]
    normalised_df_b['y'] = pca_b[:,1]
    normalised_df_b['cluster'] = labels_b
    normalised_df_b['cluster_center'] = normalised_df_b['cluster'].apply(lambda x : centers_b[x])
    
    "TNSE"
    #tsne_model = TSNE(perplexity=k, n_components=2, init='pca', method='exact',n_iter=2000, random_state=seed)
    #tsne_X_a = tsne_model.fit_transform(X_a)
    #tsne_X_b = tsne_model.fit_transform(X_b)
    #labels_a, centers_a, clusters_a = k_means(K=k, data=tsne_X_a, seed=seed)
    #labels_b, centers_b, clusters_b = k_means(K=k, data=tsne_X_b, seed=seed)
    #print_cluster(K=k, clusters=clusters, vocabulary=vocabulary, truncated=True, n=5)
    #normalised_df_a['x'] = tsne_X_a[:,0]
    #normalised_df_a['y'] = tsne_X_a[:,1]
    #normalised_df_b['x'] = tsne_X_b[:,0]
    #normalised_df_b['y'] = tsne_X_b[:,1]
    
    ##############VISUALIZATION##############
    print('Begin visualization stage')
    #set_size()
    #cluster_colors = generating_k_colors(k)
    normalised_df_a = find_concept_wordnet(k, normalised_df_a, max_depth=3)
    normalised_df_b = find_concept_wordnet(k, normalised_df_b, max_depth=3)
    """
    visualisation_normalized_word(K=k, 
              #data=tsne_X_a,
              data=pca_a,
              vocabulary=np.array(normalised_df_a['processed_text']),
              assignment=labels_a,
              colors=cluster_colors,
              plt=plt,
              title = 'Words clustering (with ontology replacement)',
              #xlim=20,
              xlim=2,
              #ylim=20)
              ylim=2)
    
    visualisation_normalized_word(K=k, 
                  #data=tsne_X_a,
                  data=pca_a,
                  vocabulary=np.array(normalised_df_a['processed_text']),
                  assignment=labels_a,
                  colors=cluster_colors,
                  plt=plt,
                  title = 'Concepts clustering (with ontology replacement)',
                  xlim=2,
                  ylim=2,
                  centralword=True,
                  df=normalised_df_a,
                  textfield = 'wordnet')
    
    visualisation_normalized_word(K=k, 
              data=tsne_X_b,
              vocabulary=np.array(normalised_df_b['processed_text']),
              assignment=labels_b,
              colors=cluster_colors,
              plt=plt,
              title = 'Words clustering (w/o ontology replacement)',
              xlim=20,
              ylim=20)
    
    visualisation_normalized_word(K=k, 
                  data=tsne_X_b,
                  vocabulary=np.array(normalised_df_b['processed_text']),
                  assignment=labels_b,
                  colors=cluster_colors,
                  plt=plt,
                  title = 'Concepts clustering (w/o ontology replacement)',
                  xlim=20,
                  ylim=20,
                  centralword=True,
                  df=normalised_df_b,
                  textfield = 'wordnet')
    """
    
    print('Finish')
    print('--------------------------------------')
    new_df.to_csv(PATHS[2],  index=False)
    replacement_df.to_csv(PATHS[3],  index=False)
    normalised_df_a.to_csv(PATHS[4],  index=False)
    normalised_df_b.to_csv(PATHS[5],  index=False)
    
    
    #return new_df, replacement_df, normalised_df_a, normalised_df_b


def update_dashboard(PATHS, textfield, fasttext, k, threshold_replace, threshold_takeout, seed):
    """
        PATHS: list of 6 paths (.csv files)
                + path 0: path to the new data. This is a csv file with header named textfield.
                + path 1: path to the processed text data (old version)
                + path 2: path to the dashboard a data (old version with remove local most frequency)
                + path 3: path to the dashboard b data (old version w/o  remove local most frequency)
                + path 4: path to the replacement
                + path 5: path to the global stopwords
    """
    
    ##############READ DATA##############
    
    new_df        = pd.read_csv(PATHS[0])
    df            = pd.read_csv(PATHS[1])
    normalised_df = pd.read_csv(PATHS[3])
    
    
    
    ##############UPDATE##############
    
    "Preprocessing new dataset"
    print('Begin preprocessing stage')
    new_df = standardize_text(new_df, textfield)
    new_df, new_normalised_df = processing_text(new_df, 'standardized_text')
    "Global Stopwords Removing"
    print('Begin global stopwords removing stage')
    english_most_frequency = pd.read_csv(PATHS[5],sep = ',', encoding = "ISO-8859-1")
    stop_words = [x for x in english_most_frequency['Word']]
    new_normalised_df['isFrequency'] = new_normalised_df['processed_text'].apply(lambda x : x in stop_words)
    new_normalised_df = new_normalised_df[new_normalised_df['isFrequency']==False]
    new_normalised_df = new_normalised_df.drop('isFrequency', axis=1)
    "Concatenate old and new data then groupby"
    print("Begin concatenate old and new data then groupby stage")
    new_df = pd.concat([df, new_df]).reset_index().drop('index', 1)
    normalised_df = normalised_df[['processed_text', 'count', 'pos']]
    
    new_normalised_df = pd.concat([normalised_df, new_normalised_df]).reset_index().drop('index', 1)
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
    "Words replacement"
    print('Begin words replacement stage')
    print('***  Before the replacement, we have {w} words'.format(w=len(new_normalised_df)))
    replacement_df, new_normalised_df = find_replacement(new_normalised_df, threshold=threshold_replace)
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
    print('***  After the replacement, we have {w} words'.format(w=len(new_normalised_df)))
    "Remove local most frequency"
    print('Begin remove local most frequency stage')
    new_normalised_df_a = new_normalised_df[new_normalised_df['count'] < threshold_takeout]
    new_normalised_df_b = new_normalised_df.copy()
    print('***  We removed {w} most frequency words'.format(w=len(new_normalised_df_b)-len(new_normalised_df_a)))
    "Clustering"
    print('Begin clustering stage')
    X_a = []
    for vect in new_normalised_df_a['vect']:
        X_a.append(list(vect))
    X_b = []
    for vect in new_normalised_df_b['vect']:
        X_b.append(list(vect))
    
    "PCA on X"
    pca = PCA(n_components=2)
    pca_a = pca.fit_transform(X_a)
    pca_b = pca.fit_transform(X_b)
    
    "Cluster"
    labels_a, centers_a, clusters_a = k_means(K=k, data=pca_a, seed=seed)
    labels_b, centers_b, clusters_b = k_means(K=k, data=pca_b, seed=seed)
    
    
    "Store the result of k-means algorithm"
    new_normalised_df_a['x'] = pca_a[:,0]
    new_normalised_df_a['y'] = pca_a[:,1]
    new_normalised_df_a['cluster'] = labels_a
    new_normalised_df_a['cluster_center'] = new_normalised_df_a['cluster'].apply(lambda x : centers_a[x])
    new_normalised_df_b['x'] = pca_b[:,0]
    new_normalised_df_b['y'] = pca_b[:,1]
    new_normalised_df_b['cluster'] = labels_b
    new_normalised_df_b['cluster_center'] = new_normalised_df_b['cluster'].apply(lambda x : centers_b[x])
    
    "Find concept"
    print('Begin find concept stage')
    cluster_colors = generating_k_colors(k)
    new_normalised_df_a = find_concept_wordnet(k, new_normalised_df_a, max_depth=3)
    new_normalised_df_b = find_concept_wordnet(k, new_normalised_df_b, max_depth=3)
    
    
    ##############STORE RESULT##############
    new_df.to_csv(PATHS[1],  index=False)
    new_normalised_df_a.to_csv(PATHS[2],  index=False)
    new_normalised_df_b.to_csv(PATHS[3],  index=False)
    replacement_df.to_csv(PATHS[4],  index=False)

def update_dashboard(PATHS, textfield, fasttext, k, threshold_replace, threshold_takeout, seed):
    """
        PATHS: list of 6 paths (.csv files)
                + path 0: path to the new data. This is a csv file with header named textfield.
                + path 1: path to the processed text data (old version)
                + path 2: path to the dashboard a data (old version with remove local most frequency)
                + path 3: path to the dashboard b data (old version w/o  remove local most frequency)
                + path 4: path to the replacement
                + path 5: path to the global stopwords
    """
    
    ##############READ DATA##############
    
    new_df        = pd.read_csv(PATHS[0])
    df            = pd.read_csv(PATHS[1])
    normalised_df = pd.read_csv(PATHS[3])
    
    
    
    ##############UPDATE##############
    
    "Preprocessing new dataset"
    print('Begin preprocessing stage')
    new_df = standardize_text(new_df, textfield)
    new_df, new_normalised_df = processing_text(new_df, 'standardized_text')
    "Global Stopwords Removing"
    print('Begin global stopwords removing stage')
    english_most_frequency = pd.read_csv(PATHS[5],sep = ',', encoding = "ISO-8859-1")
    stop_words = [x for x in english_most_frequency['Word']]
    new_normalised_df['isFrequency'] = new_normalised_df['processed_text'].apply(lambda x : x in stop_words)
    new_normalised_df = new_normalised_df[new_normalised_df['isFrequency']==False]
    new_normalised_df = new_normalised_df.drop('isFrequency', axis=1)
    "Concatenate old and new data then groupby"
    print("Begin concatenate old and new data then groupby stage")
    new_df = pd.concat([df, new_df]).reset_index().drop('index', 1)
    normalised_df = normalised_df[['processed_text', 'count', 'pos']]
    
    new_normalised_df = pd.concat([normalised_df, new_normalised_df]).reset_index().drop('index', 1)
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
    "Words replacement"
    print('Begin words replacement stage')
    print('***  Before the replacement, we have {w} words'.format(w=len(new_normalised_df)))
    replacement_df, new_normalised_df = find_replacement(new_normalised_df, threshold=threshold_replace)
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
    print('***  After the replacement, we have {w} words'.format(w=len(new_normalised_df)))
    "Remove local most frequency"
    print('Begin remove local most frequency stage')
    new_normalised_df_a = new_normalised_df[new_normalised_df['count'] < threshold_takeout]
    new_normalised_df_b = new_normalised_df.copy()
    print('***  We removed {w} most frequency words'.format(w=len(new_normalised_df_b)-len(new_normalised_df_a)))
    "Clustering"
    print('Begin clustering stage')
    X_a = []
    for vect in new_normalised_df_a['vect']:
        X_a.append(list(vect))
    X_b = []
    for vect in new_normalised_df_b['vect']:
        X_b.append(list(vect))
    
    "PCA on X"
    pca = PCA(n_components=2)
    pca_a = pca.fit_transform(X_a)
    pca_b = pca.fit_transform(X_b)
    
    "Cluster"
    labels_a, centers_a, clusters_a = k_means(K=k, data=pca_a, seed=seed)
    labels_b, centers_b, clusters_b = k_means(K=k, data=pca_b, seed=seed)
    
    
    "Store the result of k-means algorithm"
    new_normalised_df_a['x'] = pca_a[:,0]
    new_normalised_df_a['y'] = pca_a[:,1]
    new_normalised_df_a['cluster'] = labels_a
    new_normalised_df_a['cluster_center'] = new_normalised_df_a['cluster'].apply(lambda x : centers_a[x])
    new_normalised_df_b['x'] = pca_b[:,0]
    new_normalised_df_b['y'] = pca_b[:,1]
    new_normalised_df_b['cluster'] = labels_b
    new_normalised_df_b['cluster_center'] = new_normalised_df_b['cluster'].apply(lambda x : centers_b[x])
    
    "Find concept"
    print('Begin find concept stage')
    cluster_colors = generating_k_colors(k)
    new_normalised_df_a = find_concept_wordnet(k, new_normalised_df_a, max_depth=3)
    new_normalised_df_b = find_concept_wordnet(k, new_normalised_df_b, max_depth=3)
    
    
    ##############STORE RESULT##############
    new_df.to_csv(PATHS[1],  index=False)
    new_normalised_df_a.to_csv(PATHS[2],  index=False)
    new_normalised_df_b.to_csv(PATHS[3],  index=False)
    replacement_df.to_csv(PATHS[4],  index=False)

def zoom_cluster(PATH_IN, number, PATH_OUT):
    
    """
        PATH_IN: path contain dashboard data (.csv file)
        numbre : the cluster number want to zoom in
        PATH_OUT: path to write the result
    """
    
    dataframe = pd.read_csv(PATH_IN)

    new_df = dataframe[dataframe['cluster']==number]
    new_df = new_df.reset_index().drop('index', 1)

    x = []
    for vect in new_df['vect']:
        vect = vect[1:-1].split()        
        x.append([float(v) for v in vect])
    
    pca = PCA(n_components=2)
    pca_fitted = pca.fit_transform(x)
    #tsne_model = TSNE(perplexity=dataframe['cluster'].max()+1, n_components=2, init='pca', method='exact',n_iter=2000)
    #tsne_fitted = tsne_model.fit_transform(x)

    new_df['x_cluster'] = pca_fitted[:,0] #tsne_fitted[:,0]
    new_df['y_cluster'] = pca_fitted[:,1] #tsne_fitted[:,1]
    
    columns = ['processed_text', 'count', 'x_cluster', 'y_cluster', 'wordnet']
    
    new_df.to_csv(PATH_OUT, columns=columns, index=False)
    