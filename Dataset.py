import numpy as np
import pandas as pd
import random
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

'''
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
'''


def creatWordSet(dataset):
    WordSet = set([])
    for i in dataset:#list不可hash,使用遍历
        WordSet = WordSet | set(i)
    return list(WordSet)


a = ['a','b','c','a']
b = ['d','e','a','e']
c = a+b
a_1 = creatWordSet(a)
print("a_1:",a_1)

b_1 = creatWordSet(b)
print("b_1:",b_1)

c_1 = a_1 + b_1

c_2 = creatWordSet(c_1)

print("c_1:",c_1)
print("c_2:",c_2)

print("c:",c)