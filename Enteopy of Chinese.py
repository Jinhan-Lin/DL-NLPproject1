# -*- coding: utf-8 -*-
# @Date    : 2022-04-05 21:08:12
# @Author  : BrightSoul (653538096@qq.com)
import math
import os
import re
import numpy as np
import jieba
#获取语料库
def getCorpus(text_raw):
    corpus = []
    Character = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'  # 去除文字外所有字符
    ad = '本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com'  # 去除广告
    count = 0
    text_raw = re.sub(Character, '', text_raw)
    text_raw = text_raw.replace("\n", '')
    text_raw = text_raw.replace(" ", '')
    text_raw = text_raw.replace(ad, '')
    count += len(text_raw)
    corpus = text_raw.split("。")
    return corpus, count
#一元模型信息熵计算
def cal_unigram(corpus,count):
    split_words = []
    words_len = 0
    line_count = 0
    words_tf = {}
    for line in corpus:
        # for x in line #按字计算
        for x in jieba.cut(line):#按词计算
            split_words.append(x)
            words_len += 1
        get_unigram_tf(words_tf, split_words)
        split_words = []
        line_count += 1

    entropy = []
    for uni_word in words_tf.items():
        entropy.append(-(uni_word[1] / words_len) * math.log(uni_word[1] / words_len, 2))
    print("分词个数:", words_len)
    print("基于词的一元模型的中文信息熵为:", round(sum(entropy), 5), "比特/词")
    # print("基于字的一元模型的中文信息熵为:", round(sum(entropy), 5), "比特/词")
#二元模型信息熵计算
def cal_bigram(corpus, count):
    split_words = []
    words_len = 0
    line_count = 0
    words_tf = {}
    bigram_tf = {}

    for line in corpus:
        # for x in line #按字计算
        for x in jieba.cut(line):  # 按词计算
            split_words.append(x)
            words_len += 1

        get_unigram_tf(words_tf, split_words)
        get_bigram_tf(bigram_tf, split_words)

        split_words = []
        line_count += 1
    bigram_len = sum([dic[1] for dic in bigram_tf.items()])

    entropy = []
    for bi_word in bigram_tf.items():
        jp_xy = bi_word[1] / bigram_len  # 计算联合概率p(x,y)
        cp_xy = bi_word[1] / words_tf[bi_word[0][0]]  # 计算条件概率p(x|y)
        entropy.append(-jp_xy * math.log(cp_xy, 2))  # 计算二元模型的信息熵
    print("基于词的二元模型的中文信息熵为:", round(sum(entropy), 5), "比特/词")
    # print("基于字的二元模型的中文信息熵为:", round(sum(entropy), 5), "比特/词")

# 词频统计，方便计算信息熵
#一元词频统计
def get_unigram_tf(tf_dic, words):
    for i in range(len(words)-1):
        tf_dic[words[i]] = tf_dic.get(words[i], 0) + 1
# 二元词频统计
def get_bigram_tf(tf_dic, words):
    for i in range(len(words)-1):
        tf_dic[(words[i], words[i+1])] = tf_dic.get((words[i], words[i+1]), 0) + 1



if __name__ == '__main__':
    txt_path = "datasets"
    text_fileL = os.listdir(txt_path)
    for text_file in text_fileL:
        print(text_file,end=" ")
        with open(f"{txt_path}/{text_file}","r",encoding="GB18030") as fp:#读取文件
            text_raw = "".join(fp.readlines())
        corpus, count = getCorpus(text_raw)
        print("语料库字数:", count)
        cal_unigram(corpus, count)
        cal_bigram(corpus, count)
