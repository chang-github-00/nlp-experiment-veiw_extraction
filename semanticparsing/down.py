# -*- coding: utf-8 -*-
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

#词性标注
def posttagger(words):
    postagger =Postagger()#初始化实例
    postagger.load(pos_model_path) #加载模型
    postags = postagger.postag(words) #词性标注
    #for word,tag in zip(words , postags):
    #    if tag=='v':print(word + '/'+tag)
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
    print( "\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
    rely_id = [arc.head for arc in arcs]  # 提取依存父节点id
    relation = [arc.relation for arc in arcs]  # 提取依存关系
    heads = ['Root' if id == 0 else words[id - 1] for id in rely_id]  # 匹配依存父节点词语
    for i in range(len(words)):
        print(relation[i] + '(' + words[i] + ', ' + heads[i] + ')')
    parser.release()  # 释放模型

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
    return arcs

def role_label(words, postags, netags, arcs):
    labeller = SementicRoleLabeller()  # 初始化实例
    labeller.load(srl_model_path)  # 加载模型
    roles = labeller.label(words, postags, arcs)  # 语义角色标注
    for role in roles:
        print (role.index, "".join(
            ["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))
    labeller.release()  # 释放模型


def analyze_sentence(line):
    words = segmentor(line)
    postags = posttagger(words)
    netags = ner(words,postags)
    arcs=parse(words,postags)
    roles = role_label(words, postags, netags, arcs)
    words = list(words)

    tags = []
    dict = []
    for word, ntag in zip(words, netags):
        if(ntag != 'O'):#过滤非命名实体
            tags.append(ntag)
            print(word + '/' + ntag)
        if (ntag not in dict):
             dict.append(ntag)
    for tag in dict:
        num = tags.count(tag)
        print(tag + ":" + str(num))


line = "国务院港澳办严厉谴责香港极端暴徒投掷汽油弹袭警，向警察投掷汽油弹，导致警员多处烧伤"
sentences = sentence_splitter(line)
analyze_sentence(line)
#for sent in sentences:
#    analyze_sentence(sent)