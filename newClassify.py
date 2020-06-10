import numpy as np
import pandas as pd
import random
from sklearn.metrics import *
from sklearn.model_selection import train_test_split


df = pd.read_csv('/Users/macbook/Desktop/New Spam sorting.csv')
x = np.array(df['Text'].values)
y = np.array(df['Spam'].values)

#打乱数据集
index = [ i for i in range(len(df))]
random.shuffle(index)
x = x[index]
y = y[index]

# 按顺序将数据集分成10份
data_x1 = []
data_y1 = []
for i in range(10):
    data_x1.append(x[i])
    data_y1.append(y[i])
# data_x1,data_y1

data_x2 = []
data_y2 = []
for i in range(10, 20):
    data_x2.append(x[i])
    data_y2.append(y[i])

data_x3 = []
data_y3 = []
for i in range(20, 30):
    data_x3.append(x[i])
    data_y3.append(y[i])

data_x4 = []
data_y4 = []
for i in range(30, 40):
    data_x4.append(x[i])
    data_y4.append(y[i])

data_x5 = []
data_y5 = []
for i in range(40, 50):
    data_x5.append(x[i])
    data_y5.append(y[i])

data_x6 = []
data_y6 = []
for i in range(50, 60):
    data_x6.append(x[i])
    data_y6.append(y[i])

data_x7 = []
data_y7 = []
for i in range(60, 70):
    data_x7.append(x[i])
    data_y7.append(y[i])

data_x8 = []
data_y8 = []
for i in range(70, 80):
    data_x8.append(x[i])
    data_y8.append(y[i])

data_x9 = []
data_y9 = []
for i in range(80, 90):
    data_x9.append(x[i])
    data_y9.append(y[i])

datatest_x10 = []
datatest_y10 = []
for i in range(90, 100):
    datatest_x10.append(x[i])
    datatest_y10.append(y[i])


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
    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)
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
    p1Vec = np.abs(np.log(p1Num / p1allNum))
    p0Vec = np.abs(np.log(p0Num / p0allNum))
    # p1Vec = np.log(p1Num/p1allNum)
    # p0Vec = np.log(p0Num/p0allNum)
    # p1Vec = p1Num/p1allNum
    # p0Vec = p0Num/p0allNum
    return p1Vec, p0Vec


def Classify(p1Vec, p0Vec, pc1, testlist):
    p1 = sum(testlist * p1Vec) + np.log(pc1)
    p0 = sum(testlist * p0Vec) + np.log(1 - pc1)


    if np.all(p1 > p0):
        return 1
    else:
        return 0

#处理测试集数据
lower_test = creatLowerset(datatest_x10)
word_test = allword(lower_test)

#将训练集数据转化为小写形式
lower1 = creatLowerset(data_x1)
lower2 = creatLowerset(data_x2)
lower3 = creatLowerset(data_x3)
lower4 = creatLowerset(data_x4)
lower5 = creatLowerset(data_x5)
lower6 = creatLowerset(data_x6)
lower7 = creatLowerset(data_x7)
lower8 = creatLowerset(data_x8)
lower9 = creatLowerset(data_x9)

#将训练集数据分离成单词形式
word1 = allword(lower1)
word2 = allword(lower2)
word3 = allword(lower3)
word4 = allword(lower4)
word5 = allword(lower5)
word6 = allword(lower6)
word7 = allword(lower7)
word8 = allword(lower8)
word9 = allword(lower9)

#创建词汇表
wordset1 = creatWordSet(word1)
wordset2 = creatWordSet(word2)
wordset3 = creatWordSet(word3)
wordset4 = creatWordSet(word4)
wordset5 = creatWordSet(word5)
wordset6 = creatWordSet(word6)
wordset7 = creatWordSet(word7)
wordset8 = creatWordSet(word8)
wordset9 = creatWordSet(word9)

#计算先验概率
pc1_1 = ClassProbability(data_y1)
pc1_2 = ClassProbability(data_y2)
pc1_3 = ClassProbability(data_y3)
pc1_4 = ClassProbability(data_y4)
pc1_5 = ClassProbability(data_y5)
pc1_6 = ClassProbability(data_y6)
pc1_7 = ClassProbability(data_y7)
pc1_8 = ClassProbability(data_y8)
pc1_9 = ClassProbability(data_y9)

#训练集转化为词向量
veclist1 = bagOfWordVec(wordset1,word1)
veclist2 = bagOfWordVec(wordset2,word2)
veclist3 = bagOfWordVec(wordset3,word3)
veclist4 = bagOfWordVec(wordset4,word4)
veclist5 = bagOfWordVec(wordset5,word5)
veclist6 = bagOfWordVec(wordset6,word6)
veclist7 = bagOfWordVec(wordset7,word7)
veclist8 = bagOfWordVec(wordset8,word8)
veclist9 = bagOfWordVec(wordset9,word9)

#测试集转化为词向量
testVec1 = bagOfWordVec(wordset1,word_test)
testVec2 = bagOfWordVec(wordset2,word_test)
testVec3 = bagOfWordVec(wordset3,word_test)
testVec4 = bagOfWordVec(wordset4,word_test)
testVec5 = bagOfWordVec(wordset5,word_test)
testVec6 = bagOfWordVec(wordset6,word_test)
testVec7 = bagOfWordVec(wordset7,word_test)
testVec8 = bagOfWordVec(wordset8,word_test)
testVec9 = bagOfWordVec(wordset9,word_test)

#计算条件概率
xp1_1,xp0_1 = trainNB(veclist1,data_y1)
xp1_2,xp0_2 = trainNB(veclist2,data_y2)
xp1_3,xp0_3 = trainNB(veclist3,data_y3)
xp1_4,xp0_4 = trainNB(veclist4,data_y4)
xp1_5,xp0_5 = trainNB(veclist5,data_y5)
xp1_6,xp0_6 = trainNB(veclist6,data_y6)
xp1_7,xp0_7 = trainNB(veclist7,data_y7)
xp1_8,xp0_8 = trainNB(veclist8,data_y8)
xp1_9,xp0_9 = trainNB(veclist9,data_y9)

'''
def testing(xp1,xp0,pc1,testVec,data_y):
    y_predict = []
    for i in range(len(testVec)):
        y_predict.append(Classify(xp1,xp0,pc1,testVec))

    #accuracy = accuracy_score(y_test, y_predict)
    return accuracy_score(data_y, y_predict)

accuracy1 = testing(xp1_1,xp0_1,pc1_1,testVec1,data_y1)
'''



def testing1():
    y_predict = []
    for i in range(len(testVec1)):
        y_predict.append(Classify(xp1_1,xp0_1,pc1_1,testVec1))

    #accuracy = accuracy_score(y_test, y_predict)
    return accuracy_score(data_y1, y_predict)



def testing2():
    y_predict = []
    for i in range(len(testVec2)):
        y_predict.append(Classify(xp1_2,xp0_2,pc1_2,testVec2))

    #accuracy = accuracy_score(y_test, y_predict)
    return accuracy_score(data_y2, y_predict)


def testing3():
    y_predict = []
    for i in range(len(testVec3)):
        y_predict.append(Classify(xp1_3,xp0_3,pc1_3,testVec3))

    #accuracy = accuracy_score(y_test, y_predict)
    return accuracy_score(data_y3, y_predict)


def testing4():
    y_predict = []
    for i in range(len(testVec4)):
        y_predict.append(Classify(xp1_4,xp0_4,pc1_4,testVec4))

    #accuracy = accuracy_score(y_test, y_predict)
    return accuracy_score(data_y4, y_predict)


def testing5():
    y_predict = []
    for i in range(len(testVec5)):
        y_predict.append(Classify(xp1_5,xp0_5,pc1_5,testVec5))

    #accuracy = accuracy_score(y_test, y_predict)
    return accuracy_score(data_y5, y_predict)


def testing6():
    y_predict = []
    for i in range(len(testVec6)):
        y_predict.append(Classify(xp1_6,xp0_6,pc1_6,testVec6))

    #accuracy = accuracy_score(y_test, y_predict)
    return accuracy_score(data_y6, y_predict)


def testing7():
    y_predict = []
    for i in range(len(testVec7)):
        y_predict.append(Classify(xp1_7,xp0_7,pc1_7,testVec7))

    #accuracy = accuracy_score(y_test, y_predict)
    return accuracy_score(data_y7, y_predict)


def testing8():
    y_predict = []
    for i in range(len(testVec8)):
        y_predict.append(Classify(xp1_8,xp0_8,pc1_8,testVec8))

    #accuracy = accuracy_score(y_test, y_predict)
    return accuracy_score(data_y8, y_predict)


def testing9():
    y_predict = []
    for i in range(len(testVec9)):
        y_predict.append(Classify(xp1_9,xp0_9,pc1_9,testVec9))

    #accuracy = accuracy_score(y_test, y_predict)
    return accuracy_score(data_y9, y_predict)

accuracy1 = testing1()
accuracy2 = testing2()
accuracy3 = testing3()
accuracy4 = testing4()
accuracy5 = testing5()
accuracy6 = testing6()
accuracy7 = testing7()
accuracy8 = testing8()
accuracy9 = testing9()

accuracy = (accuracy1+accuracy2+accuracy3+accuracy4+accuracy5+accuracy6+accuracy7+accuracy8+accuracy9)/9
print(accuracy)
