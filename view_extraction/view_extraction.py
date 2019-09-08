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
import gensim
import os
import jieba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.linalg import svd
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import AgglomerativeClustering
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
    #for se in list_sent:
    #   print(se,'\n')
    return list(sents)

#预处理，提取出需要细致分句的内容

#分词
def segmentor(sentence):
    segmentor = Segmentor()#初始化实例
    segmentor.load(cws_model_path)
    words = segmentor.segment(sentence)
    #print('|'.join(words))
    words_list = list(words)
    segmentor.release() #释放模型
    return words

#词性标注
def posttagger(words):
    postagger =Postagger()#初始化实例
    postagger.load(pos_model_path) #加载模型
    postags = postagger.postag(words) #词性标注
    postagger.release()
    return postags


#命名实体识别
def ner(words,postags):
    recognizer = NamedEntityRecognizer() #初始化实例
    recognizer.load(ner_model_path) #加载模型
    netags = recognizer.recognize(words,postags) #命名实体识别
    #for word ,ntag in zip(words,netags):
    #    print(word + '/'+ntag)
    recognizer.release()
    return netags

def parse(words, postags):
    parser = Parser() # 初始化实例
    parser.load(par_model_path)  # 加载模型
    arcs = parser.parse(words, postags)  # 句法分析
    #print( "\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
    parser.release()  # 释放模型
    '''
    # 利用networkx绘制句法分析结果
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['font.family'] = 'sans-serif'
    G = nx.Graph()  # 建立无向图G
    # 添加节点
    for word in words:
        G.add_node(word)
    G.add_node('Root')
    # 添加边
    for i in range(len(words)):
        G.add_edge(words[i], heads[i])

    source = '国务院'
    target1 = '谴责'
    distance1 = nx.shortest_path_length(G, source=source, target=target1)
    print("'%s'与'%s'在依存句法分析图中的最短距离为:  %s" % (source, target1, distance1))
    nx.draw(G, with_labels=True)
    plt.savefig("undirected_graph.png")
    '''
    return arcs

def role_label(words, postags, netags, arcs):
    labeller = SementicRoleLabeller()  # 初始化实例
    labeller.load(srl_model_path)  # 加载模型
    roles = labeller.label(words, postags, arcs)  # 语义角色标注
    #for role in roles:
    #   print (role.index, "".join(   ["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))
    labeller.release()  # 释放模型
    return roles

def find_start_say(line,sayword):
    words = segmentor(line)
    postags = posttagger(words)
    netags = ner(words, postags)
    arcs = parse(words, postags)
    roles=role_label(words,postags,netags,arcs)
    saywordindex=999
    for i in range(len(words)):
        if words[i] == sayword:
            saywordindex=i
            break
    for role in roles:
        if role.index==saywordindex:
            for arg in role.arguments:
                if arg.name=="A1":
                    startpoint=line.index(words[arg.range.start])
                    return startpoint
    startpoint=line.index(sayword)+len(sayword)+1
    while line[startpoint]==' 'or line[startpoint]==','or line[startpoint]==':':
        startpoint=startpoint+1
    return  startpoint
def find_whom(line,sayword):
    words = segmentor(line)
    postags = posttagger(words)
    netags = ner(words, postags)
    arcs = parse(words, postags)
    rely_id = [arc.head for arc in arcs]  # 提取依存父节点id
    relation = [arc.relation for arc in arcs]  # 提取依存关系
    heads = ['Root' if id == 0 else words[id - 1] for id in rely_id]  # 匹配依存父节点词语
    for i in range(len(words)):
        if relation[i]=='SBV':
            if words[rely_id[i]-1]==sayword:
                return words[i]

filename='most_similar_words.txt'
def get_similar_words(filename):
    sayset=set(["说"])
    with open(filename, 'r',encoding='UTF-8') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines: break
            for i in lines.split(' '):
                sayset=sayset|set([i])
    return sayset

def analyze_verb(line,sayset):
    words = segmentor(line)
    postags = posttagger(words)
    for word, tag in zip(words, postags):
        if tag == 'v':
            #print(word + '/' + tag)
            if word in sayset:
                #print("view extraction begins")
                return word
    return "not found"

def findner(verb):
    words = segmentor(line)
    postags = posttagger(words)
    netags = ner(words, postags)
    arcs = parse(words, postags)

def lines2mat(lines):
    model = gensim.models.doc2vec.Doc2Vec.load("my_doc2vec_model")
    mat=[]
    sentences = sentence_splitter(lines)
    '''
    vocabSet = set([])
    for sent in sentences:
        words = segmentor(sent)
        vocabSet = vocabSet | set(words)
    #print(vocabSet,'\n')
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
    '''
    for sent in sentences:
        words = segmentor(sent)
        vector = model.infer_vector(words, alpha=0.1, min_alpha=0.0001, steps=20)
        mat.append(list(vector))
    #print('mat', mat)
    return mat

def lsa(list_lines): #得到svd后文段单位化的句子向量
    mat = np.array(list_lines)
    U, D, V = svd(mat)
    #print('U',U)
    y = np.linalg.norm(U, axis=1, keepdims=True)
    #print(y)
    U = U / y
    #print('3-D\n', U)
    return U

def k_means(normalized_lines):
    getnum=np.shape(normalized_lines)[0]
    estimator = KMeans(n_clusters=getnum-1)  # 构造聚类器
    estimator.fit(normalized_lines)  # 聚类
    label_pred = estimator.labels_  # 获取聚类标签
    return label_pred

def findpause(labels):
    cmp=9999
    note=99
    i=1
    all=np.shape(np.array(labels))
    for i in range(1,all[0]-1):
        var1=np.var(labels[:i])
        var2=np.var(labels[i:all[0]])
        vartotal=max(var1,var2)
        if vartotal<cmp :note=i
    return note


#主要处理函数
def get_parse(doc):
    docs=sentence_splitter(doc)
    sayset=get_similar_words(filename)
    sayword=analyze_verb(doc,sayset)
    if sayword=="not found": return "no view embedded"
    else:
        st=0
        for i in docs:
            if i.find(sayword)!=-1:
                ed=st+1
                for j in docs[st+1:]:
                    if analyze_verb(j,sayset)!="not found":
                        break
                    ed = ed + 1
                break
            st=st+1
        startparse="".join(docs[st:ed])
        who = find_whom(startparse, sayword)
        if who==None:
            return startparse,who
        num=find_start_say(startparse,sayword)
        startparse=startparse[num:]
        for i in range(len(startparse)):
            if startparse[i] == '“':
                for j in range(i, len(startparse)):
                    if startparse[j] == '”':
                        return startparse[i:j],who
        if ed-st>1:
            sentences = sentence_splitter(startparse)
            list_lines = lines2mat(startparse)
            new_lines = lsa(list_lines)
            #print('new lines \n', new_lines)
            labels = k_means(new_lines)
            #print ('labels', labels)
            end = findpause(labels)
            return "".join(sentences[:end]),who
        else:
            return startparse,who

def analyze_report(report):
    print("report:", report)
    rest = report
    while rest != "":
        toget, bywhom = get_parse(rest)
        if bywhom!=None:
            print("view by", bywhom, ":", toget)
        num = report.index(toget) + len(toget)
        rest = report[num:]
if __name__ == '__main__':
    report1="法国总统马克龙26日在G7峰会期间宣布，七国集团(G7)国家已同意为一项2000万美元的计划提供资金，以帮助扑灭亚马孙森林大火。马克龙还表示，尽管特朗普没有出席有关环境问题的领导人会议，但美国支持这项计划。但白宫国家安全委员会发言人加勒特马奎斯28日却表示，美国并未承诺提供援助。马奎斯在一份声明中说：“美国随时准备帮助巴西应对这些火灾，但不同意七国集团的一项联合倡议，该倡议没有包括与巴西总统的磋商。”“最具建设性的援助方式是与巴西政府进行协调。"
    analyze_report(report1)
    report2="据俄罗斯卫星通讯社7日报道，俄总统新闻局称，“双方正面评价互相释放并移交两国境内关押人员的做法。并重点强调这种做法在人道主义方面起的作用,对于双边关系正常化有着重要意义”。当天，俄罗斯和乌克兰进行了大规模囚犯交换。此举被认为两国关系解冻的一个信号。消息指出，双方领导人在电话中讨论了乌克兰内部调解的问题，两国领导人强调遵守停火制度以及逐步从接触线撤出军队和武器的重要性。"
    analyze_report(report2)
'''
sentences = sentence_splitter(report)
list_lines=lines2mat(report)
new_lines=lsa(list_lines)
print('new lines \n',new_lines)
labels=k_means(new_lines)
print ('labels',labels)
end=findpause(labels)
print(' '.join(sentences[:end]))
'''