import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv('/Users/macbook/Desktop/New Spam sorting.csv')
x = df['Text'].values
y = df['Spam'].values
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
#将邮件内容转化为小写形式
def creatLowerset(dataset):
    lower_word = []
    for i in dataset:
        lower_word.append(i.lower())
    return lower_word
lower_X_train = creatLowerset(X_train)
lower_X_test = creatLowerset(X_test)
#将文本内容分离成单词形式
def allword(dataset):
    word = []
    for i in dataset:
        word.append(i.split( ))
    return word
word_X_train = allword(lower_X_train)
word_X_test = allword(lower_X_test)
#创建词汇表（元素唯一不重复）
def creatWordSet(dataset):
    WordSet = set([])
    for i in dataset:#list不可hash,使用遍历
        WordSet = WordSet | set(i)
    return list(WordSet)
WordList_X_train = creatWordSet(word_X_train)
WordList_X_test = creatWordSet(word_X_test)
#特征提取
#使用sklearn计算tfidf值特征
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer, TfidfVectorizer
corpus = X_train
vectorizer=CountVectorizer()#构建一个计算词频（TF）
transformer = TfidfTransformer()# #构建一个计算TF-IDF
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus)) #将文本corpus输入，得到词频矩阵
print(tfidf)


def ClassProbability(dataset):
    spam_num = 0
    pro_spam = 0

    for i in dataset:
        if i == 1:
            spam_num += 1
    pro_spam = spam_num / len(y_train)
    return pro_spam
pc1 = ClassProbability(y_train)
pc0 = 1-pc1
def bagOfWordVec(wordlist,dataset):
    VecList = []
    for document in dataset:
        returnVec = [0]*len(wordlist)
        for word in document:
            if word in wordlist:
                returnVec[wordlist.index(word)] += 1
        VecList.append(returnVec)
    return VecList
trainVec = bagOfWordVec(WordList_X_train,word_X_train)
testVec = bagOfWordVec(WordList_X_train,word_X_test)

def trainNB(trainMatrix,trainCategory):
    numWords = len(trainMatrix[0])#每条样本的词条数量
    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)
    p0allNum = 0.0
    p1allNum = 0.0
    for i in range(len(trainMatrix)):#样本数量
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]#如果是垃圾邮件，就把该索引位置下的的样本文件对应的向量形式相加
            p1allNum += sum(trainMatrix[i])#把所有垃圾邮件中，所有词频相加
        else:
            p0Num += trainMatrix[i]
            p0allNum += sum(trainMatrix[i])
        p1Vec = log(p1Num/P1allNum)
        p0Vec = log(p0Num/P0allNum)
        return p1Vec,p0Vec
    x_p1, x_p0 = trainNB(WordList_X_train, y_train)

def Classify(p1Vec, p0Vec, pc1, testVec):
    p1 = sum(trainVec * p1Vec) + log(pc1)
    p0 = sum(trainVec * p0Vec) + log(1 - pc1)
    if p1 > p0:
        return 1
    else:
        return 0

def testing():
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(X_test)):
            if Classify(p1Vec, p0Vec, pc1, testVec[i]) == 1 & y_test[i] == 1:
                    TP += 1
            if Classify(p1Vec, p0Vec, pc1, testVec[i]) == 1 & y_test[i] == 0:
                    FP += 1
            if Classify(p1Vec, p0Vec, pc1, testVec[i]) == 0 & y_test[i] == 0:
                    TN += 1
            if Classify(p1Vec, p0Vec, pc1, testVec[i]) == 0 & y_test[i] == 1:
                    FN += 1
    testing()
    (TP + TN) / (len(X_test))