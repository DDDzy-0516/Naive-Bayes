#速度快，准确率低

import numpy as np
import pandas as pd
import random
from sklearn.metrics import *
from sklearn.model_selection import train_test_split


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
print("y的长度：",num)


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

    n = 0
    WordSet = set([])
    for i in dataset:#list不可hash,使用遍历
        WordSet = WordSet | set(i)
        n += 1
    print("n:",n)
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



#将训练集数据转化为小写形式
lower1 = creatLowerset(data_x1)
lower2 = creatLowerset(data_x2)
lower3 = creatLowerset(data_x3)
lower4 = creatLowerset(data_x4)
print("小写：")
print(len(lower1))
print(len(lower2))
print(len(lower3))
print(len(lower4))


#将训练集数据分离成单词形式
word1 = allword(lower1)
word2 = allword(lower2)
word3 = allword(lower3)
word4 = allword(lower4)
print("单词形式：")
print(len(word1))
print(len(word2))
print(len(word3))
print(len(word4))



#创建词汇表
wordset1 = creatWordSet(word1)
wordset2 = creatWordSet(word2)
wordset3 = creatWordSet(word3)
wordset4 = creatWordSet(word4)



#测试集为1
X_test1 = word1
X_train1 = word2 + word3 + word4
Y_test1 = data_y1
Y_train1 = data_y2 + data_y3 + data_y4

print("-------------------")
print(len(wordset2))
print(len(wordset3))
print(len(wordset4))
print("-------------------")

vocabset1 = wordset2 + wordset3 + wordset4

print("vocabset1:", len(vocabset1))

vocabulary1 = creatWordSet(vocabset1)

print("vocablary1:", len(vocabulary1))

pc1_1 = ClassProbability(Y_train1)

print("pc1:",pc1_1)

train_vec1 = bagOfWordVec(vocabulary1,X_train1)
test_vec1 = bagOfWordVec(vocabulary1,X_test1)

xp1_1,xp0_1 = trainNB(train_vec1,Y_train1)

def testing1():
    y_predict = []
    for i in range(len(test_vec1)):
        y_predict.append(Classify(xp1_1,xp0_1, pc1_1, test_vec1[i]))

    # accuracy = accuracy_score(y_test, y_predict)
    print("1:", y_predict)
    print("1:", data_y1)
    return accuracy_score(data_y1, y_predict)




#测试集为2
X_test2 = word2
X_train2 = word1 + word3 + word4
Y_test2 = data_y2
Y_train2 = data_y1 + data_y3 + data_y4
vocabset2 = wordset1 + wordset3 + wordset4
vocabulary2 = creatWordSet(vocabset2)

print("vocablary2:", len(vocabulary2))

pc1_2 = ClassProbability(Y_train2)

print("pc2:",pc1_2)

train_vec2 = bagOfWordVec(vocabulary2,X_train2)
test_vec2 = bagOfWordVec(vocabulary2,X_test2)

xp1_2,xp0_2 = trainNB(train_vec2,Y_train2)

def testing2():
    y_predict = []
    for i in range(len(test_vec2)):
        y_predict.append(Classify(xp1_2,xp0_2, pc1_2, test_vec2[i]))

    # accuracy = accuracy_score(y_test, y_predict)
    print("2:", y_predict)
    print("2:", data_y2)
    return accuracy_score(data_y2, y_predict)




#测试集为3
X_test3 = word3
X_train3 = word1 + word2 + word4
Y_test3 = data_y3
Y_train3 = data_y1 + data_y2 + data_y4
vocabset3 = wordset1 + wordset2 + wordset4
vocabulary3 = creatWordSet(vocabset3)

print("vocablary3:", len(vocabulary3))

pc1_3 = ClassProbability(Y_train3)

print("pc3:",pc1_3)

train_vec3 = bagOfWordVec(vocabulary3,X_train3)
test_vec3 = bagOfWordVec(vocabulary3,X_test3)

xp1_3,xp0_3 = trainNB(train_vec3,Y_train3)

def testing3():
    y_predict = []
    for i in range(len(test_vec3)):
        y_predict.append(Classify(xp1_3,xp0_3, pc1_3, test_vec3[i]))

    # accuracy = accuracy_score(y_test, y_predict)
    print("3:", y_predict)
    print("3:", data_y3)
    return accuracy_score(data_y3, y_predict)




#测试集为4
X_test4 = word4
X_train4 = word1 + word2 + word3
Y_test4 = data_y4
Y_train4 = data_y1 + data_y2 + data_y3
vocabset4 = wordset1 + wordset2 + wordset3
vocabulary4 = creatWordSet(vocabset4)

print("vocablary4:", len(vocabulary4))

pc1_4 = ClassProbability(Y_train4)

print("pc4:",pc1_4)

train_vec4 = bagOfWordVec(vocabulary4,X_train4)
test_vec4 = bagOfWordVec(vocabulary4,X_test4)

xp1_4,xp0_4 = trainNB(train_vec4,Y_train4)

def testing4():
    y_predict = []
    for i in range(len(test_vec4)):
        y_predict.append(Classify(xp1_4,xp0_4, pc1_4, test_vec4[i]))

    # accuracy = accuracy_score(y_test, y_predict)
    print("4:", y_predict)
    print("4:", data_y4)
    return accuracy_score(data_y4, y_predict)



accuracy1 = testing1()
accuracy2 = testing2()
accuracy3 = testing3()
accuracy4 = testing4()

accuracy = (accuracy1 + accuracy2 + accuracy3 + accuracy4)/4

print("------------------------------------")

print("accuracy1:", accuracy1)
print("accuracy2:", accuracy2)
print("accuracy3:", accuracy3)
print("accuracy4:", accuracy4)

print("------------------------------------")

print("accuracy:", accuracy)
