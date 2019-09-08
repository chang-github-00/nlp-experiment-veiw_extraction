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
cws_model_path = os.path.join(LTP_DATA_DIR,'cws.model')#�ִ�
par_model_path = os.path.join(LTP_DATA_DIR,'parser.model')#����ʶ��
ner_model_path = os.path.join(LTP_DATA_DIR,'ner.model')#����ʵ��ʶ��
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')# ���Ա�עģ��·����ģ������Ϊ`pos.model`
srl_model_path = os.path.join(LTP_DATA_DIR,'pisrl_win.model')
#�־䣬���ı���ɾ���
def sentence_splitter(sentence):
    sents = SentenceSplitter.split(sentence) #�־�
    list_sent=list(sents)
    #for se in list_sent:
    #   print(se,'\n')
    return list(sents)

#Ԥ������ȡ����Ҫϸ�·־������

#�ִ�
def segmentor(sentence):
    segmentor = Segmentor()#��ʼ��ʵ��
    segmentor.load(cws_model_path)
    words = segmentor.segment(sentence)
    #print('|'.join(words))
    words_list = list(words)
    segmentor.release() #�ͷ�ģ��
    return words

#���Ա�ע
def posttagger(words):
    postagger =Postagger()#��ʼ��ʵ��
    postagger.load(pos_model_path) #����ģ��
    postags = postagger.postag(words) #���Ա�ע
    postagger.release()
    return postags


#����ʵ��ʶ��
def ner(words,postags):
    recognizer = NamedEntityRecognizer() #��ʼ��ʵ��
    recognizer.load(ner_model_path) #����ģ��
    netags = recognizer.recognize(words,postags) #����ʵ��ʶ��
    #for word ,ntag in zip(words,netags):
    #    print(word + '/'+ntag)
    recognizer.release()
    return netags

def parse(words, postags):
    parser = Parser() # ��ʼ��ʵ��
    parser.load(par_model_path)  # ����ģ��
    arcs = parser.parse(words, postags)  # �䷨����
    #print( "\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
    parser.release()  # �ͷ�ģ��
    '''
    # ����networkx���ƾ䷨�������
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['font.family'] = 'sans-serif'
    G = nx.Graph()  # ��������ͼG
    # ��ӽڵ�
    for word in words:
        G.add_node(word)
    G.add_node('Root')
    # ��ӱ�
    for i in range(len(words)):
        G.add_edge(words[i], heads[i])

    source = '����Ժ'
    target1 = 'Ǵ��'
    distance1 = nx.shortest_path_length(G, source=source, target=target1)
    print("'%s'��'%s'������䷨����ͼ�е���̾���Ϊ:  %s" % (source, target1, distance1))
    nx.draw(G, with_labels=True)
    plt.savefig("undirected_graph.png")
    '''
    return arcs

def role_label(words, postags, netags, arcs):
    labeller = SementicRoleLabeller()  # ��ʼ��ʵ��
    labeller.load(srl_model_path)  # ����ģ��
    roles = labeller.label(words, postags, arcs)  # �����ɫ��ע
    #for role in roles:
    #   print (role.index, "".join(   ["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))
    labeller.release()  # �ͷ�ģ��
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
    rely_id = [arc.head for arc in arcs]  # ��ȡ���游�ڵ�id
    relation = [arc.relation for arc in arcs]  # ��ȡ�����ϵ
    heads = ['Root' if id == 0 else words[id - 1] for id in rely_id]  # ƥ�����游�ڵ����
    for i in range(len(words)):
        if relation[i]=='SBV':
            if words[rely_id[i]-1]==sayword:
                return words[i]

filename='most_similar_words.txt'
def get_similar_words(filename):
    sayset=set(["˵"])
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
    '''
    for sent in sentences:
        words = segmentor(sent)
        vector = model.infer_vector(words, alpha=0.1, min_alpha=0.0001, steps=20)
        mat.append(list(vector))
    #print('mat', mat)
    return mat

def lsa(list_lines): #�õ�svd���Ķε�λ���ľ�������
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
    estimator = KMeans(n_clusters=getnum-1)  # ���������
    estimator.fit(normalized_lines)  # ����
    label_pred = estimator.labels_  # ��ȡ�����ǩ
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


#��Ҫ������
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
            if startparse[i] == '��':
                for j in range(i, len(startparse)):
                    if startparse[j] == '��':
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
    report1="������ͳ�����26����G7����ڼ��������߹�����(G7)������ͬ��Ϊһ��2000����Ԫ�ļƻ��ṩ�ʽ��԰�������������ɭ�ִ�����������ʾ������������û�г�ϯ�йػ���������쵼�˻��飬������֧������ƻ������׹����Ұ�ȫίԱ�ᷢ���˼��������˹28��ȴ��ʾ��������δ��ŵ�ṩԮ�������˹��һ��������˵����������ʱ׼����������Ӧ����Щ���֣�����ͬ���߹����ŵ�һ�����ϳ��飬�ó���û�а����������ͳ�Ĵ��̡�������߽����Ե�Ԯ����ʽ���������������Э����"
    analyze_report(report1)
    report2="�ݶ���˹����ͨѶ��7�ձ���������ͳ���žֳƣ���˫���������ۻ����ͷŲ��ƽ��������ڹ�Ѻ��Ա�����������ص�ǿ�������������˵����巽���������,����˫�߹�ϵ������������Ҫ���塱�����죬����˹���ڿ��������˴��ģ�����������˾ٱ���Ϊ������ϵ�ⶳ��һ���źš���Ϣָ����˫���쵼���ڵ绰���������ڿ����ڲ���������⣬�����쵼��ǿ������ͣ���ƶ��Լ��𲽴ӽӴ��߳������Ӻ���������Ҫ�ԡ�"
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