import spacy
from sklearn.cluster import KMeans
from operatorUtils import defineOperator
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import gensim
import time
from dashboard import get_dashboard, update_dashboard, zoom_cluster
from lsa import check_responses

nlp = spacy.load('en_core_web_lg')

operatorPackage = {
    'id': 'op-jigsaw-ext',
    'type': 'social',
    'external': True,
    'outputDefinition': ['group', 'role'],
    'meta': {
        'name': 'Jigsaw external operator',
        'shortDesc': 'Generates roles and groups',
        'description': 'Written in Python'},
    'config': {
        'type': 'object',
        'properties': {
            'roles': {
                'type': 'string',
                'title': 'List roles, separated by comma'
            }
        }
    }
}


##################################GLOBAL VARIABLES##################################

"The first 3 files are already in folder data, however we have to download fasttext pre-trained model"
PATH_TO_STOPWORD = "data/english_most_frequency.csv"
POS_JAR = "data/stanford-postagger-full-2017-06-09/stanford-postagger-3.8.0.jar"
POS_MODEL = "data/stanford-postagger-full-2017-06-09/models/french.tagger"
PATH_TO_FASTTEXT = ".../wiki.en/wiki.en.vec"


THRESHOLD_REPLACEMENT = 0.7 #this threshold is the similarity score between two words
THRESHOLD_TAKEOUT = 50 #local frequency, if a word was present more than this threshold then we remove it
NB_CLUSTER = 40
SEED = 9016891
TEXT_FIELD = "text"
FASTTEXT = gensim.models.KeyedVectors.load_word2vec_format(PATH_TO_FASTTEXT)
english_most_5000_frequency = pd.read_csv(PATH_TO_STOPWORD,sep = ',', encoding = "ISO-8859-1")
STOPWORDS = [x for x in english_most_5000_frequency['Word']]
TEACHER_ANSWER = " "
LATENT_SPACE = 60

responseScore_df = None
similarity_matrix = None
dashboard_df = None
processedText_df = None
replacement_df = None
zoom_df = None


#dashboard = {}
#words = {}

def newData(data, dashboardId):

    # Write to global variable
    global dashboard_df
    global processedText_df
    global replacement_df

    studentmapping = [k for k, v in object['activityData']['payload'].items()]

    "Read texts from students"
    texts = [v['data']['text'] for k, v in object['activityData']['payload'].items()]

    "Transform to dataframe"
    new_df = pd.DataFrame(texts, columns=[TEXT_FIELD])

    "Process new data"
    processedText_df, replacement_df, dashboard_df = update_dashboard(new_df, 
                                                                      TEXT_FIELD,
                                                                      processedText_df,
                                                                      replacement_df,
                                                                      dashboard_df,
                                                                      FASTTEXT, 
                                                                      STOPWORDS, 
                                                                      NB_CLUSTER, 
                                                                      THRESHOLD_REPLACEMENT, 
                                                                      SEED)


    ######EXPORT NEEDED DATA FROM GLOBAL VARIABLE DATAFRAME######
    #processedText_df.to_json()
    #replacement_df.to_json()
    #dashboard_df.to_json()


"""
  dashboardChoice can be either `dasbhoard1` or `dasbhoard2` or `a number` 
"""
def getDashboard(dashboardChoice , dashboardId):

    global dashboard_df
    global processedText_df
    global replacement_df
    global zoom_df


    studentmapping = [k for k, v in object['activityData']['payload'].items()]
    "Read texts from students"
    texts = [v['data']['text'] for k, v in object['activityData']['payload'].items()]

    "Create dataframe"
    new_df = pd.DataFrame(texts, columns=[TEXT_FIELD])

    if (dashboardChoice == 'dasbhoard1'):
        processedText_df, replacement_df, dashboard_df = get_dashboard(new_df, 
                                                                       TEXT_FIELD, 
                                                                       FASTTEXT, 
                                                                       STOPWORDS, 
                                                                       NB_CLUSTER, 
                                                                       THRESHOLD_REPLACEMENT, 
                                                                       SEED)
    elif (dashboardChoice == 'dasbhoard2'):
        processedText_df, replacement_df, dashboard_df = get_dashboard(new_df, 
                                                                       TEXT_FIELD, 
                                                                       FASTTEXT, 
                                                                       STOPWORDS, 
                                                                       NB_CLUSTER, 
                                                                       THRESHOLD_REPLACEMENT, 
                                                                       SEED,
                                                                       True,
                                                                       THRESHOLD_TAKEOUT)
    else:
        nbCluster = int(cluster)
        zoom_df = zoom_cluster(dashboard_df, nbCluster)
    


    ######EXPORT NEEDED DATA FROM GLOBAL VARIABLE DATAFRAME######

    #processedText_df.to_json()
    #replacement_df.to_json()
    #dashboard_df.to_json()









"""
   data can be in english or french
"""
def getGroupStudent(english=True):

  global responseScore_df
  global similarity_matrix

  "Read texts from students"
  texts = [v['data']['text'] for k, v in object['activityData']['payload'].items()]

  "Create dataframe"
  new_df = pd.DataFrame(texts, columns=[TEXT_FIELD])

  if english:
    responseScore_df, similarity_matrix = check_responses(new_df, TEXT_FIELD, LATENT_SPACE, TEACHER_ANSWER)
  else:
    responseScore_df, similarity_matrix = check_responses(new_df, TEXT_FIELD, LATENT_SPACE, TEACHER_ANSWER, POS_JAR, POS_MODEL, english=english)

  ######EXPORT NEEDED DATA FROM GLOBAL VARIABLE DATAFRAME######
  #responseScore_df.to_json()
  #similarity_matrix is a numpy.ndarray

defineOperator(operatorPackage, operator)