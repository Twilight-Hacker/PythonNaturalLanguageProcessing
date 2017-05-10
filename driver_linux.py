import os
import re
import random
import unicodedata
import pandas as pd
from sklearn.linear_model.stochastic_gradient import SGDClassifier as sdg
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


train_path = "../resource/asnlib/public/aclImdb/train/" # use terminal to ls files under this directory
test_path = "../resource/asnlib/public/imdb_te.csv" # test data for grade evaluation

def imdb_data_preprocess(inpath, outpath="./", name="imdb_tr.csv", mix=False):
    pos_path = inpath + "pos/"
    neg_path = inpath + "neg/"
    pos_text_files = [str(pos_path + f) for f in os.listdir(pos_path) if f.endswith('.txt')]
    neg_text_files = [str(neg_path + f) for f in os.listdir(neg_path) if f.endswith('.txt')]
    
    train_out = open(name, 'w')
    
    for fi_name in pos_text_files:
        fi = open(fi_name)
        text = '"' + fi.read() + '"'
        text = re.sub('<.*\/?>', '', text)
        text = re.sub(',', ' ', text)
        text = re.sub('\n', '.', text)
        pol = "1"
        res = text + "," + pol + "\n"
        train_out.write(res)
        
    for fi_name in neg_text_files:
        fi = open(fi_name)
        text = '"' + fi.read() + '"'
        text = re.sub('<.*\/?>', '', text)
        text = re.sub(',', ' ', text)
        text = re.sub('\n', '.', text)
        pol = "0"
        res = text + "," + pol +"\n"
        train_out.write(res)
        
    train_out.close()
    
    #REMOVE THIS PART IF THE PROJECT FAILS HORRIBLY IN PREDICTION
    with open(name,'r') as train_out:
        data = [ (random.random(), line) for line in train_out ]
    data.sort()
    
    i = 0
    
    with open(name,'w') as train_out:
        for _, line in data:
            line = str(i) +"," + line
            train_out.write( line )
            i += 1


nfout_uni_nm = "unigram.output.txt"
nfout_uni_tf = "unigramtfidf.output.txt"
nfout_bi_nm = "bigram.output.txt"
nfout_bi_tf = "bigramtfidf.output.txt"


if __name__ == "__main__":
    imdb_data_preprocess(train_path)
    
    stpwords = []
    
    with open("stopwords.en.txt") as filel:
        for w in filel:
            w = re.sub('\n?', '', w)
            stpwords.append(w)
            
    textData = []
    y = []
    
    with open("imdb_tr.csv") as f:
        for li in f:
            temp = [i for i in li.split(",")]
            #temp[0] is the index
            t = temp[1]
            t = unicode(t, 'utf-8')
            t = unicodedata.normalize('NFKD', t).encode('utf-8', 'ignore')
            textData.append(str(t))
            y.append(int(temp[2]))
    
    
    
    monogram = CountVectorizer(ngram_range=(1,1), stop_words=stpwords)
    bigram = CountVectorizer(ngram_range=(1,2), stop_words=stpwords) #range (1,2) decided by discussion
    tdif_monogram = TfidfVectorizer(ngram_range=(1,1), stop_words=stpwords)
    tdif_bigram = TfidfVectorizer(ngram_range=(1,2), stop_words=stpwords) #range (1,2) decided by discussion
    
    Tmonogram = monogram.fit_transform(textData, y)
    Tbigram = bigram.fit_transform(textData, y)
    Ttdif_monogram = tdif_monogram.fit_transform(textData, y)
    Ttdif_bigram = tdif_bigram.fit_transform(textData, y)
    
    model1 = sdg(loss="hinge", penalty='l1')
    model2 = sdg(loss="hinge", penalty='l1')
    model3 = sdg(loss="hinge", penalty='l1')
    model4 = sdg(loss="hinge", penalty='l1')
    
    Mmonogram = model1.fit(Tmonogram, y)
    Mbigram = model2.fit(Tbigram, y)
    Mtdif_monogram = model3.fit(Ttdif_monogram, y)
    Mtdif_bigram = model4.fit(Ttdif_bigram, y)
            
    testData = []
    
    df=pd.read_csv(test_path, sep=",", header=None,encoding = 'ISO-8859-1')
    
    for i in range(1,len(df[1])):
        testData.append(df[1][i])

    fout_uni_nm = open(nfout_uni_nm, 'w')
    fout_uni_tf = open(nfout_uni_tf, 'w')
    fout_bi_nm = open(nfout_bi_nm, 'w')
    fout_bi_tf = open(nfout_bi_tf, 'w')
    
    UTmonogram = monogram.transform(testData)
    UTbigram = bigram.transform(testData)
    UTtdif_monogram = tdif_monogram.transform(testData)
    UTtdif_bigram = tdif_bigram.transform(testData)
    
    pred_mon_s = Mmonogram.predict(UTmonogram)
    pred_bi_s = Mbigram.predict(UTbigram)
    pred_mon_t = Mtdif_monogram.predict(UTtdif_monogram)
    pred_bi_t = Mtdif_bigram.predict(UTtdif_bigram)
    
    for i in range(len(pred_mon_s)):
        fout_uni_nm.write(str(pred_mon_s[i])+"\n")
        fout_uni_tf.write(str(pred_mon_t[i])+"\n")
        fout_bi_nm.write(str(pred_bi_s[i])+"\n")
        fout_bi_tf.write(str(pred_bi_t[i])+"\n")
        
        
    
    fout_uni_nm.close()
    fout_uni_tf.close()
    fout_bi_nm.close()
    fout_bi_tf.close()
    
