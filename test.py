

import numpy as np
import pandas as pd
import random



#df = pd.read_csv('/Users/macbook/Desktop/New Spam sorting.csv')
df = pd.read_csv('/Users/macbook/Desktop/Spam sorting.csv')
x = np.array(df['text'].values)
y = np.array(df['spam'].values)


print("共有样本：",len(y),"条")

spamNum = 0

for i in range(len(y)):
    if y[i] == 1:
        spamNum += 1

print("垃圾邮件所占比例为：",spamNum/len(y))

num = len(y)
print(num)


#打乱数据集
index = [ i for i in range(len(df))]
random.shuffle(index)
x = x[index]
y = y[index]


# 按顺序将数据集分成4份,共5728条数据，每份为1432条
data_x1 = []
data_y1 = []
for i in range(0,int(num/4)):
    data_x1.append(x[i])
    data_y1.append(y[i])
# data_x1,data_y1

data_x2 = []
data_y2 = []
for i in range(int(num/4),int((num/4)*2)):
    data_x2.append(x[i])
    data_y2.append(y[i])

data_x3 = []
data_y3 = []
for i in range(int((num/4)*2),int((num/4)*3)):
    data_x3.append(x[i])
    data_y3.append(y[i])

data_x4 = []
data_y4 = []
for i in range((int((num/4)*3)),num):
    data_x4.append(x[i])
    data_y4.append(y[i])



#将邮件内容转化为小写形式
def creatLowerset(dataset):
    lower_word = []
    for i in dataset:
        lower_word.append(i.lower())
    return lower_word

#将文本内容分离成单词形式
def allword(dataset):
    word = []
    for i in dataset:
        word.append(i.split( ))
    return word

#创建词汇表（元素唯一不重复）
def creatWordSet(dataset):
    WordSet = set([])
    for i in dataset:#list不可hash,使用遍历
        WordSet = WordSet | set(i)
    return list(WordSet)


# 计算先验概率
def ClassProbability(dataset):
    spam_num = 0
    pro_spam = 0

    for i in dataset:
        if i == 1:
            spam_num += 1
    pro_spam = spam_num / len(dataset)
    return pro_spam


# 词向量
def bagOfWordVec(wordlist, dataset):
    VecList = []
    for document in dataset:
        returnVec = [0] * len(wordlist)
        for word in document:
            if word in wordlist:
                returnVec[wordlist.index(word)] += 1
        VecList.append(returnVec)

    return VecList


def trainNB(trainMatrix, trainCategory):
    # print(trainMatrix[1])
    numWords = len(trainMatrix[0])  # 每条样本的词条数量
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    # print(len(p0Num))
    p0allNum = 0.0
    p1allNum = 0.0
    for i in range(len(trainMatrix)):  # 样本数量

        if trainCategory[i] == 1:
            # print(trainMatrix[i], type(trainMatrix[i]))
            p1Num += trainMatrix[i]
            # 如果是垃圾邮件，就把该索引位置下的的样本文件对应的向量形式相加
            p1allNum += sum(trainMatrix[i])  # 把所有垃圾邮件中，所有词频相加
        else:
            p0Num += trainMatrix[i]
            p0allNum += sum(trainMatrix[i])
    # print(p1allNum,p0allNum)
    # epsilon = 1e-5
    p1Vec = np.log(p1Num / p1allNum)
    p0Vec = np.log(p0Num / p0allNum)
    #p1Vec = np.log(p1Num) - np.log(p1allNum)
    #p0Vec = np.log(p0Num) - np.log(p0allNum)
    return p1Vec, p0Vec


def Classify(p1Vec, p0Vec, pc1, testlist):
    #print(len(testlist))
    #print(len(p1Vec))
    '''
    print(type(testlist))
    print(type(p1Vec))
    print(type(p0Vec))
    print(type(pc1))
    print(pc1)
    '''
    #print(testlist[0])
    p1 = sum(testlist * p1Vec) + np.log(pc1)
    p0 = sum(testlist * p0Vec) + np.log(1 - pc1)

    #print(type(p1))
    #print("p1:",p1)
    #print("p0:",p0)

    if p1 > p0:
        return 1
    else:
        return 0

def evaluation(y_test, y_predict):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_test)):
        if y_test[i] == 1 and y_predict[i] == 1:
            TP += 1
        if y_test[i] == 1 and y_predict[i] == 0:
            FP += 1
        if y_test[i] == 0 and y_predict[i] == 0:
            TN += 1
        if y_test[i] == 0 and y_predict[i] == 1:
            FN += 1

    print(TP, FP, TN, FN)
    # 准确率Accuracy
    accuracy = (TP + TN) / len(y_test)

    # 精确率Precision
    precision = TP / (TP + FP)

    # 召回率recall
    recall = TP / (TP + FN)

    # F-score
    F1 = (2 * TP) / (2 * TP + FP + FN)

    return accuracy, precision, recall, F1


#data_x1,data_y1作为测试集
def testing1():

    X_train = []
    X_train = data_x2 + data_x3 + data_x4
    Y_train = []
    Y_train = data_y2 + data_y3 + data_y4


    #将测试集和训练集转化成小写形式
    lower_test = creatLowerset(data_x1)
    lower_train = creatLowerset(X_train)

    #将测试集和训练集文本内容分离成单词形式
    word_test = allword(lower_test)
    word_train = allword(lower_train)

    #创建训练集词汇表
    vocabulary = creatWordSet(word_train)
    print("vocabulary 1:",len(vocabulary))

    #计算先验概率
    pc1 = ClassProbability(Y_train)
    print("pc1:",pc1)

    #将测试集和训练集转化为词向量
    test_vec = bagOfWordVec(vocabulary, word_test)
    train_vec = bagOfWordVec(vocabulary, word_train)

    #计算条件概率
    xp1, xp0 = trainNB(train_vec, Y_train)

    y_predict = []
    for i in range(len(test_vec)):
        y_predict.append(Classify(xp1, xp0, pc1, test_vec[i]))

    # accuracy = accuracy_score(y_test, y_predict)
    print("1:",y_predict)
    print("1:",data_y1)

    accuracy, precision, recall, F1 = evaluation(data_y1, y_predict)

    return accuracy, precision, recall, F1


def testing2():

    X_train = []
    X_train = data_x1 + data_x3 + data_x4
    Y_train = []
    Y_train = data_y1 + data_y3 + data_y4


    #将测试集和训练集转化成小写形式
    lower_test = creatLowerset(data_x2)
    lower_train = creatLowerset(X_train)

    #将测试集和训练集文本内容分离成单词形式
    word_test = allword(lower_test)
    word_train = allword(lower_train)

    #创建训练集词汇表
    vocabulary = creatWordSet(word_train)
    print("vocabulary 2:", len(vocabulary))

    #计算先验概率
    pc1 = ClassProbability(Y_train)
    print("pc2:", pc1)

    #将测试集和训练集转化为词向量
    test_vec = bagOfWordVec(vocabulary, word_test)
    train_vec = bagOfWordVec(vocabulary, word_train)

    #计算条件概率
    xp1, xp0 = trainNB(train_vec, Y_train)

    y_predict = []
    for i in range(len(test_vec)):
        y_predict.append(Classify(xp1, xp0, pc1, test_vec[i]))

    # accuracy = accuracy_score(y_test, y_predict)
    print("2:",y_predict)
    print("2:",data_y2)

    accuracy, precision, recall, F1 = evaluation(data_y2, y_predict)

    return accuracy, precision, recall, F1


def testing3():

    X_train = []
    X_train = data_x1 + data_x2 + data_x4
    Y_train = []
    Y_train = data_y1 + data_y2 + data_y4


    #将测试集和训练集转化成小写形式
    lower_test = creatLowerset(data_x3)
    lower_train = creatLowerset(X_train)

    #将测试集和训练集文本内容分离成单词形式
    word_test = allword(lower_test)
    word_train = allword(lower_train)

    #创建训练集词汇表
    vocabulary = creatWordSet(word_train)
    print("vocabulary 3:", len(vocabulary))

    #计算先验概率
    pc1 = ClassProbability(Y_train)
    print("pc3:", pc1)

    #将测试集和训练集转化为词向量
    test_vec = bagOfWordVec(vocabulary, word_test)
    train_vec = bagOfWordVec(vocabulary, word_train)

    print("''''''''''''''")
    #print("测试集词向量3：",test_vec)

    #print("特征集词向量3：", train_vec)

    #计算条件概率
    xp1, xp0 = trainNB(train_vec, Y_train)

    #print("xp1_3:",xp1)
    #print("xp0_3:", xp0)

    y_predict = []
    for i in range(len(test_vec)):
        y_predict.append(Classify(xp1, xp0, pc1, test_vec[i]))

    # accuracy = accuracy_score(y_test, y_predict)
    print("3:",y_predict)
    print("3:",data_y3)

    accuracy, precision, recall, F1 = evaluation(data_y3, y_predict)

    return accuracy, precision, recall, F1



def testing4():

    X_train = []
    X_train = data_x2 + data_x3 + data_x1
    Y_train = []
    Y_train = data_y2 + data_y3 + data_y1


    #将测试集和训练集转化成小写形式
    lower_test = creatLowerset(data_x4)
    lower_train = creatLowerset(X_train)

    #将测试集和训练集文本内容分离成单词形式
    word_test = allword(lower_test)
    word_train = allword(lower_train)

    #创建训练集词汇表
    vocabulary = creatWordSet(word_train)
    print("vocabulary 4:", len(vocabulary))

    #计算先验概率
    pc1 = ClassProbability(Y_train)
    print("pc4:", pc1)

    #将测试集和训练集转化为词向量
    test_vec = bagOfWordVec(vocabulary, word_test)
    train_vec = bagOfWordVec(vocabulary, word_train)

    #计算条件概率
    xp1, xp0 = trainNB(train_vec, Y_train)

    y_predict = []
    for i in range(len(test_vec)):
        y_predict.append(Classify(xp1, xp0, pc1, test_vec[i]))

    # accuracy = accuracy_score(y_test, y_predict)
    print("4:",y_predict)
    print("4:",data_y4)

    accuracy, precision, recall, F1 = evaluation(data_y4, y_predict)

    return accuracy, precision, recall, F1






accuracy1, precision1, recall1, F11 = testing1()
accuracy2, precision2, recall2, F12 = testing2()
accuracy3, precision3, recall3, F13 = testing3()
accuracy4, precision4, recall4, F14 = testing4()

print("1:",accuracy1, precision1, recall1, F11)
print("2:",accuracy2, precision2, recall2, F12)
print("3:",accuracy3, precision3, recall3, F13)
print("4:",accuracy4, precision4, recall4, F14)

accuracy = (accuracy1 + accuracy2 + accuracy3 + accuracy4)/4
precision = (precision1 + precision2 + precision3 + precision4)/4
recall = (recall1 + recall2 + recall3 + recall4)/4
F1 = (F11 + F12 + F13 + F14)/4

print("平均accuracy:", accuracy)
print("平均precision:",precision)
print("平均recall:", recall)
print("平均F1:",F1)