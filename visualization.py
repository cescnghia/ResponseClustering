# -*- coding: utf-8 -*-
import numpy as np


def visualisation_normalized_word(K, data, vocabulary, assignment, colors, plt, title, xlim, ylim, centralword=False, df=None, textfield=None):
    """
        K: number of cluster
        data: 2D array contains x,y coordinates
        vocabulary: words' labels
        assignment: cluster assignment
        colors: set of colors for cluster
        centralword: True if we plot only central word, otherwise plot all words
        df: dataframe contains needed information for computing cluster center
    """
    
    plt.figure(figsize=(30, 30))
    plt.xlim((-xlim,xlim))
    plt.ylim((-ylim,ylim))
    plt.title(title)
    
    
    x = data[:,0]
    y = data[:,1]

    for i in range(len(x)):
            plt.scatter(x[i],y[i], c=colors[assignment[i]])
    

    if centralword:

        df = df[df['representative_cluster']==True]
        for idx, row in df.iterrows():
            plt.annotate(row[textfield],
                     xy=(row['cluster_center'][0], row['cluster_center'][1]),
                     textcoords='offset points',
                     ha='right',
                     va='bottom'
                    )

    else:
        for i in range(len(x)):
            if (i % 3 != 0 or i%4 !=0):
                plt.annotate(vocabulary[i],
                             xy=(x[i], y[i]),
                             xytext=(5, 2),
                             textcoords='offset points',
                             ha='right',
                             va='bottom'
                            )
        
    plt.show()