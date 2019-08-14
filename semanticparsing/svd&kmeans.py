# coding=gbk
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import Parser
from pyltp import NamedEntityRecognizer
from pyltp import SementicRoleLabeller
from pyltp import SentenceSplitter
from pyltp import SementicRoleLabeller
from graphviz import Digraph
import networkx as nx
import matplotlib.pyplot as plt
from pylab import mpl
import os
import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.linalg import svd
LTP_DATA_DIR ="C:\\Users\\machang\\ltp_data_v3.4.0"
cws_model_path = os.path.join(LTP_DATA_DIR,'cws.model')#�ִ�
par_model_path = os.path.join(LTP_DATA_DIR,'parser.model')#����ʶ��
ner_model_path = os.path.join(LTP_DATA_DIR,'ner.model')#����ʵ��ʶ��
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')# ���Ա�עģ��·����ģ������Ϊ`pos.model`
srl_model_path = os.path.join(LTP_DATA_DIR,'pisrl_win.model')
#�־䣬���ı���ɾ���
def sentence_splitter(sentence):
    sents = SentenceSplitter.split(sentence) #�־�
    list_sent=list(sents)
    for se in list_sent:
        print(se,'\n')
    return list(sents)

#�ִ�
def segmentor(sentence):
    segmentor = Segmentor()#��ʼ��ʵ��
    segmentor.load(cws_model_path)
    words = segmentor.segment(sentence)
    print('|'.join(words))
    words_list = list(words)
    segmentor.release() #�ͷ�ģ��
    return words
def lines2mat(lines):
    mat=[]
    sentences = sentence_splitter(lines)
    vocabSet = set([])
    for sent in sentences:
        words = segmentor(sent)
        vocabSet = vocabSet | set(words)
    print(vocabSet,'\n')
    vocabList=list(vocabSet)
    for sent in sentences:
        returnVec = [0] * len(vocabList)  # ����һ����������Ԫ�ض�Ϊ0������
        wordss = segmentor(sent)
        for word in set(wordss):
            if word in vocabList:
                # returnVec[vocabList.index(word)] = 1     # index�������ַ������ҵ��ַ���һ�γ��ֵ�λ��  �ʼ�ģ��
                temp = vocabList.index(word)
                # print(temp,'\n')
                returnVec[vocabList.index(word)] += 1  # �ĵ��Ĵʴ�ģ��    ÿ�����ʿ��Գ��ֶ��
            else:
                x = 1
                # print ("the word: %s is not in my Vocabulary!" % word)
        not mat.append(returnVec)
    return mat
def lsa(list_lines): #�õ�svd���Ķε�λ���ľ�������
    mat=np.array(list_lines)
    U,D,V=svd(mat)
    U=np.array(U[:,:2])
    print('2-D\n', U)
    y = np.linalg.norm(U, axis=1, keepdims=True)
    print(y)
    U = U / y
    print('2-D\n', U)
    return U
def k_means(normalized_lines):
    X=normalized_lines
    estimator = KMeans(n_clusters=2)  # ���������
    estimator.fit(X)#����
    label_pred = estimator.labels_ #��ȡ�����ǩ
    #����k-means���
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.scatter(0, 0, c="blue", marker='+')
    x0 = X[label_pred == 0]
    x1 = X[label_pred == 1]
    plt.scatter(x0[:, 0], x0[:, 1], c = "red", marker='o', label='label0')
    plt.scatter(x1[:, 0], x1[:, 1], c = "green", marker='*', label='label1')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc=2)
    plt.show()

report="����ʿ����԰�⣬һЩ����԰����������԰��չ���ݵ�ȷʵҲ�С���ֹЯ��ʳƷ������԰�������ƹ涨��������Ϊ�����ο���֪���Ѿ�ע����ؽ�ֹ�Թ涨���ο��ڹ�����Ʊ����԰ʱ����ζ��ѡ�������һ�涨�����������Ϊ��һ�۵���Щ���ԡ���ȫ�ž��ο��Դ�ʳƷ������԰�����ܻ��һЩ��Ҫʳ������ʳ����οʹ������㣬�����顣�����Լ�������ʳƷ���ϣ���û����ɰ�ȫ��������û��Ӱ�컷��������ȴ�޷���԰����ʳ�ã����������ȹ���԰����ָ����ʳƷ���ϡ��Ҽ۸�߳�԰��֮�⣬Υ����������Ȩ�汣�������ڹ�ƽ���׺ͽ�ֹǿ�����ѵĹ涨���ַ��������ߺϷ�Ȩ�棬���Ϸ�������԰������Ա�Ƿ���Զ��οͽ���ǿ�Ƽ�飬����Ȼ�Ƿ񶨵ġ��ҹ��ܷ���������涨�������������ɲ����ַ�����ֹ�Ƿ��Ѳ鹫�����壻�������������Ȩ�����Ʋ�Ȩ���ܷ��ɱ������κ���֯���߸��˲����ַ����������԰������Ա��Ϊȷʵ��Ҫ���м�飬ҲӦ���������οͱ���ͬ�⣬�ɷ�����Ա��顣"
list_lines=lines2mat(report)
new_lines=lsa(list_lines)
k_means(new_lines)