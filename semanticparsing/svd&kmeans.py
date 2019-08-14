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
cws_model_path = os.path.join(LTP_DATA_DIR,'cws.model')#分词
par_model_path = os.path.join(LTP_DATA_DIR,'parser.model')#依存识别
ner_model_path = os.path.join(LTP_DATA_DIR,'ner.model')#命名实体识别
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')# 词性标注模型路径，模型名称为`pos.model`
srl_model_path = os.path.join(LTP_DATA_DIR,'pisrl_win.model')
#分句，把文本变成句子
def sentence_splitter(sentence):
    sents = SentenceSplitter.split(sentence) #分句
    list_sent=list(sents)
    for se in list_sent:
        print(se,'\n')
    return list(sents)

#分词
def segmentor(sentence):
    segmentor = Segmentor()#初始化实例
    segmentor.load(cws_model_path)
    words = segmentor.segment(sentence)
    print('|'.join(words))
    words_list = list(words)
    segmentor.release() #释放模型
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
        returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为0的向量
        wordss = segmentor(sent)
        for word in set(wordss):
            if word in vocabList:
                # returnVec[vocabList.index(word)] = 1     # index函数在字符串里找到字符第一次出现的位置  词集模型
                temp = vocabList.index(word)
                # print(temp,'\n')
                returnVec[vocabList.index(word)] += 1  # 文档的词袋模型    每个单词可以出现多次
            else:
                x = 1
                # print ("the word: %s is not in my Vocabulary!" % word)
        not mat.append(returnVec)
    return mat
def lsa(list_lines): #得到svd后文段单位化的句子向量
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
    estimator = KMeans(n_clusters=2)  # 构造聚类器
    estimator.fit(X)#聚类
    label_pred = estimator.labels_ #获取聚类标签
    #绘制k-means结果
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

report="除迪士尼乐园外，一些动物园、主题游乐园、展览馆等确实也有“禁止携带食品饮料入园”的类似规定。有人认为，《游客须知》已经注明相关禁止性规定，游客在购买门票和入园时即意味着选择接受这一规定。反对意见认为这一观点有些绝对。完全杜绝游客自带食品饮料入园，可能会给一些需要食用特殊食物的游客带来不便，不合情。明明自己购买了食品饮料，既没有造成安全隐患，又没有影响环境卫生，却无法在园区内食用，不合理。被迫购买园区内指定的食品饮料、且价格高出园区之外，违反了消费者权益保护法关于公平交易和禁止强制消费的规定，侵犯了消费者合法权益，不合法。游乐园工作人员是否可以对游客进行强制检查，答案显然是否定的。我国宪法和民法总则规定，公民人身自由不受侵犯，禁止非法搜查公民身体；民事主体的人身权利、财产权利受法律保护，任何组织或者个人不得侵犯。如果游乐园工作人员认为确实需要进行检查，也应当在征得游客本人同意，由法定人员检查。"
list_lines=lines2mat(report)
new_lines=lsa(list_lines)
k_means(new_lines)