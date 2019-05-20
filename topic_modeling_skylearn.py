import nltk
import re
import sys
import os
import numpy as np
import datetime as time
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

STOPWORDS = nltk.corpus.stopwords.words('russian')
NUM_TOPICS = 10
WITH_PRINT = True

def main():
    fls = readFiles()
    print('Find %s text groups', len(fls))  
    for f in fls:
        lda_str_tuples, nmf_str_tuples, lsi_str_tuples = topic_process(f['texts'])
        f['lda_str_tuples'] = lda_str_tuples
        f['nmf_str_tuples'] = nmf_str_tuples
        f['lsi_str_tuples'] = lsi_str_tuples
        print("Create topic for group: %s, files: %s", f['group'], f['files'])
    createReport(fls)

def readFiles():
    groupFolders = list(filter(lambda g: g.is_dir(), os.scandir('texts')))
    rows = []
    for d in groupFolders:
        groupName = d.name
        textFiles = list(filter(lambda f: f.name.endswith('.txt'), os.scandir(d.path)))
        texts = []
        files = []
        for path in list(map(lambda f: f.path, textFiles)):
            with open(path, 'r') as f:
                text = ' ' + f.read().replace('\xad', ' ').replace('\xa0', ' ').replace('\n', ' ')
                texts.append(text)
                files.append(f.name)
        rows.append({'group': groupName, 'texts': texts, 'files': files}) 
    return rows

def transform_topics_to_str(model, vectorizer, top_n=10):
    topics = []
    for idx, topic in enumerate(model.components_):
        topic_num_str = str.format("Topic %d:" % (idx)) 
        topic_str = [(vectorizer.get_feature_names()[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]] 
        topics.append([topic_num_str, topic_str])
    return topics

def topic_process(text):
    vectorizer = TfidfVectorizer(min_df=1, max_df=0.9, stop_words=STOPWORDS, lowercase=True)
    data_vectorized = vectorizer.fit_transform(text)
    
    # Build a Latent Dirichlet Allocation Model
    lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
    lda_Z = lda_model.fit_transform(data_vectorized)
    # print(lda_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
    
    # Build a Non-Negative Matrix Factorization Model
    nmf_model = NMF(n_components=NUM_TOPICS)
    nmf_Z = nmf_model.fit_transform(data_vectorized)
    # print(nmf_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
    
    # Build a Latent Semantic Indexing Model
    lsi_model = TruncatedSVD(n_components=NUM_TOPICS)
    lsi_Z = lsi_model.fit_transform(data_vectorized)
    # print(lsi_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
    
    
    # Let's see how the first document in the corpus looks like in different topic spaces
    lda_str_tuples = transform_topics_to_str(lda_model, vectorizer)
    print_topics("LDA Model:", lda_str_tuples)
    # print(lda_Z[0])
    
    nmf_str_tuples = transform_topics_to_str(nmf_model, vectorizer)
    print_topics("NMF Model:", nmf_str_tuples)
    # print(nmf_Z[0])
    
    lsi_str_tuples = transform_topics_to_str(lsi_model, vectorizer)
    print_topics("LSI Model:", lsi_str_tuples)
    # print(lsi_Z[0])
    
    return lda_str_tuples, nmf_str_tuples, lsi_str_tuples

def print_topics(title, topic_rows):
    if WITH_PRINT:
        print(title)
        for t in topic_rows:
            print(t)    

def getCurrentDate():
    return time.datetime.today().strftime('%Y_%m_%d_%H_%M_%S')

def createReport(rows):
    f = open('out/report' + getCurrentDate() + '.txt', 'w+')
    for r in rows:
        f.write('Group: ' + r['group'])
        f.write('\n')
        f.write('LDA Model:')
        f.write('\n')
        for m in r['lda_str_tuples']:
            f.write(m[0])
            f.write('[' + ', '.join(list(map(lambda x: x[0], m[1]))) + ']')
            f.write('\n')
        f.write('NMF Model:')
        f.write('\n')
        for m in r['nmf_str_tuples']:
            f.write(m[0])
            f.write('[' + ', '.join(list(map(lambda x: x[0], m[1]))) + ']')
            f.write('\n')
        f.write('LSI Model:')
        f.write('\n')
        for m in r['lsi_str_tuples']:
            f.write(m[0])
            f.write('[' + ', '.join(list(map(lambda x: x[0], m[1]))) + ']')
            f.write('\n')
        f.write('\n')
        f.write('=' * 20)
        f.write('\n')
    f.close()

main()
