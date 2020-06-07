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
        word.append(i.split())
    return word

word_X_train = allword(lower_X_train)
word_X_test = allword(lower_X_test)
#print(word_X_train)

#创建词汇表（元素唯一不重复）
def creatWordSet(dataset):
    WordSet = set([])
    for i in dataset:
        WordSet = WordSet | set(i)
    return list(WordSet)

WordList_X_train = creatWordSet(word_X_train)
#print(WordList_X_train)

def ClassProbability(dataset):
    spam_num = 0
    pro_spam = 0

    for i in dataset:
        if i == 1:
            spam_num +=1
    pro_spam = spam_num/len(y_train)
    return pro_spam

pc1 = ClassProbability(y_train)

def bagofWordVec(wordlist,dataset):
    VecList = []
    for document in dataset:
        returnVec = [0]*len(wordlist)
        for word in document:
            if word in wordlist:
                returnVec[wordlist.index(word)] += 1
        VecList.append(returnVec)
    return VecList

trainVec = bagofWordVec(WordList_X_train,word_X_train)
testVec = bagofWordVec(WordList_X_train,word_X_test)
#print(trainVec)

def trainNB(trainMatrix,trainCategory):
    #numWords = len(trainMatrix[0])
    #print(trainMatrix)

    p0Num = np.zeros(len(trainMatrix[1]))
    p1Num = np.zeros(len(trainMatrix[1]))

    p0allNum = 0
    p1allNum = 0

    for i in range(len(trainMatrix)):

        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1allNum += sum(trainMatrix[i])

        else:
            p0Num += trainMatrix[i]
            p0allNum += sum(trainMatrix[i])

    #print(p1allNum,p0allNum)
    #print(np.log(p1Num/p1allNum))
    epsilon = 1e-5
    p1Vec = np.abs(np.log(p1Num/p1allNum + epsilon))
    p0Vec = np.abs(np.log(p0Num/p0allNum + epsilon))

    return p1Vec,p0Vec



#trainNB(trainVec,y_train)
x_p1,x_p0 = trainNB(trainVec,y_train)

'''
where_are_inf1 = np.isinf(x_p1)
where_are_inf0 = np.isinf(x_p0)
x_p1[where_are_inf1] = 0
x_p0[where_are_inf0] = 0
'''

def Classify(p1Vec,p0Vec,pc1,testlist):
    p1 = sum(testlist*p1Vec) + np.log(pc1)
    p0 = sum(testlist * p1Vec) + np.log(1-pc1)

    if p1 > p0:
        return 1
    else:
        return 0

def testing():
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(testVec)):
        if Classify(x_p1,x_p0,pc1,testVec[i]) == 1 and y_test[i] == 1:
            TP+=1
        if Classify(x_p1,x_p0,pc1,testVec[i]) == 1 and y_test[i] == 0:
            FP+=1
        if Classify(x_p1,x_p0,pc1,testVec[i]) == 0 and y_test[i] == 0:
            TN+=1
        if Classify(x_p1,x_p0,pc1,testVec[i]) == 0 and y_test[i] == 1:
            FN+=1

    return TP,FP,TN,FN

def realnum(y_test):
    Numclass1 = 0
    for i in range(len(y_test)):
        if y_test[i] == 1:
            Numclass1 += 1

    print(Numclass1/len(y_test),Numclass1)

TP,FP,TN,FN = testing()
realnum(y_test)

print(TP,FP,TN,FN,(TP+TN)/(len(X_test)))