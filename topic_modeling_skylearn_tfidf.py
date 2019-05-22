import nltk
import re
import sys
import os
import numpy as np
import datetime as time
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from pymystem3 import Mystem
from string import punctuation, digits

STOPWORDS = []
NUM_TOPICS = 10

WITH_PRINT = False
ONLY_NOUNS = True

punctuation = set(punctuation + '«»—–…“”\n\t' + digits)
TABLE = str.maketrans({ch: ' ' for ch in punctuation})
mapping = {'COM': 'ADJ', 'APRO': 'DET', 'PART': 'PART', 'PR': 'ADP', 'ADV': 'ADV', 'INTJ': 'INTJ',
           'S': 'NOUN', 'V': 'VERB', 'CONJ': 'SCONJ', 'UNKN': 'X', 'ANUM': 'ADJ', 'NUM': 'NUM',
           'NONLEX': 'X', 'SPRO': 'PRON', 'ADVPRO': 'ADV', 'A': 'ADJ'}
pymystem = Mystem()

def main():
    gr_fls = readFiles()
    fls_len = len(gr_fls)
    print('Find {} text groups'.format(fls_len))  
    cur_len = 0
    for f in gr_fls:
        reset_stopwords()
        txt_list = f['texts']
        target_stopwords = STOPWORDS
        if ONLY_NOUNS == True:
            joined_text = ' '.join(txt_list)
            lws = lemmatize_words(pymystem, joined_text, mapping)
            target_stopwords = extract_nouns(lws)
        f['stopwords'] = target_stopwords
        #lemmatize process
        lemmatized_texts = lemmatized_process(txt_list)
        #tokenize process
        lda_str_tuples, nmf_str_tuples, lsi_str_tuples = topic_process(lemmatized_texts, target_stopwords)
        f['lda_str_tuples'] = lda_str_tuples
        f['nmf_str_tuples'] = nmf_str_tuples
        f['lsi_str_tuples'] = lsi_str_tuples
        cur_len += 1
        print('Proccess by {} from {} ...'.format(cur_len, fls_len))
    print("Create report...")
    createReport(gr_fls)
    print("Finish")

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

def reset_stopwords():
    STOPWORDS = nltk.corpus.stopwords.words('russian')

def lemmatize_one(txt):
    lemmas = pymystem.lemmatize(txt)
    return ''.join(lemmas)

def lemmatized_process(txt_list):
    return list(map(lambda t: lemmatize_one(t), txt_list))

def lemmatize_words(pymystem, text, mapping):
    lemmas_pos = []
    ana = pymystem.analyze(text.translate(TABLE))
    for word in ana:
        if word.get('analysis') and len(word.get('analysis')) > 0:
            lemma = word['analysis'][0]['lex'].lower().strip()
            if lemma not in STOPWORDS:
                pos = word['analysis'][0]['gr'].split(',')[0]
                pos = pos.split('=')[0].strip()
                if pos in mapping:
                    lemmas_pos.append([lemma, mapping[pos]])
                else:
                    lemmas_pos.append([lemma, '_X']) # на случай, если попадется тэг, которого нет в маппинге
    return lemmas_pos

def extract_nouns(lemmatized_words):
    all_stop_words = STOPWORDS.copy()
    for lt in lemmatized_words:
        wtp = lt[1].lower().strip()
        if (wtp == 'noun' or wtp == '_x') is False:
            if WITH_PRINT == True:
                print("Add in stops " + lt[0] + ' with ' + lt[1] + " wtp " + wtp)
            all_stop_words.append(lt[0])
    return all_stop_words

def transform_topics_to_str(model, vectorizer, top_n=10):
    topics = []
    for idx, topic in enumerate(model.components_):
        topic_num_str = str.format("Topic %d:" % (idx)) 
        topic_str = [(vectorizer.get_feature_names()[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]] 
        topics.append([topic_num_str, topic_str])
    return topics

def topic_process(texts_vec, target_stop_words):
    vectorizer = TfidfVectorizer(min_df=1, max_df=0.9, stop_words=target_stop_words, lowercase=True)
    data_vectorized = vectorizer.fit_transform(texts_vec)
    
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
    if WITH_PRINT == True:
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
        if WITH_PRINT == True:
            f.write('STOPWORDS:\n')
            f.write(', '.join(r['stopwords']))
        f.write('=' * 20)
        f.write('\n')
    f.close()

main()