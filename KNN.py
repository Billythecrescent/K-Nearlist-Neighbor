#KNN algorithm for titanic.dat
from sys import argv
from random import randint
from math import *

#1. Read the titanic.dat file
def readDat(filename):
    f = open(filename,'r')
    lines = f.readlines()
    pclass,age,sex = [],[],[]
    people = []
    for i in lines:
        if i[0] != '@':
            values = i.split(',')
            pclass.append(float(values[0]))
            age.append(float(values[1]))
            sex.append(float(values[2]))
            attributes = map(float,values[:-1])
            label = float(values[-1][:-1]) #remove the \n
            person = {"attribute":attributes,"label":label}
            people.append(person)
    return people,pclass,age,sex
    #print(people)
    #print(len(people))

#readDat("titanic.dat")

#2. Standardise the data using Range method, to [0,1]
def ReadplusStandard(filename):
    people,pclass,age,sex = readDat(filename)
    maxclass,minclass = max(pclass),min(pclass)
    maxage,minage = max(age),min(age)
    maxsex,minsex = max(sex),min(sex)
    standclass = lambda x: (x-minclass)/(maxclass-minclass)
    standage = lambda x: (x-minage)/(maxage-minage)
    standsex = lambda x: (x-minsex)/(maxsex-minsex)
    for person in people:
        a,b,c = person["attribute"]
        person["attribute"] = standclass(a),standage(b),standsex(c)
    return people
    #print(people)

#ReadplusStandard("titanic.dat")

#3. Divide the data into two sets, with the ratio of 7:3
#people: the standardlised data structure
#ratio: ratio of training data and test data, from 0 to 1
def DataDevide(people,ratio):
    randomintlist = []
    traindata = []
    testdata = []
    for i in range(int(floor(len(people)*ratio))):
        index = randint(0,len(people)-1)
        while index in randomintlist:
            index = randint(0,len(people)-1)
        randomintlist.append(index)
        traindata.append(people[index])
    for i in range(len(people)-1):
        if i not in randomintlist:
            testdata.append(people[i])
    return traindata,testdata
    #print(len(traindata))
    #print(len(testdata))

#a,b are both 3-dimension list, [class,age,sex]
def distance(a,b):
    distance = sqrt(sum(map((lambda x,y: (x-y)**2),a,b)))
    return distance


#4. Classify the data of the test dataset
#traindata: list[diary{"attribute":list[],"label":int}]
#case: diary{"attribute":list[],"label":int}
#use Mahalanobis Distance
def KNNClassify(traindata,case,k):
    size = len(traindata)
    for i in traindata:
        dis = distance(i["attribute"],case["attribute"])
        i["distance"] = dis
    sortdata = sorted(traindata,key = lambda data: data["distance"])
    #print(sortdata)
    referlist = sortdata[:k]
    classification = {}
    total_num = 0
    for i in referlist:
        if i["label"] in classification:
            classification[i["label"]]+=1
        else:
            classification[i["label"]]=1
    #print(classification)
    result = sorted(classification.items(),key = lambda data: data[1])
    return result[-1][0],result[-1][1]/k
    #result: [(-1.0, 4), (1.0, 46)]

def ClassTest(traindata,testdata,k):
    n = len(testdata)
    result = []
    for case in testdata:
        test = KNNClassify(traindata,case,k)
        label = case["label"]
        temp = {"test":test,"label":label}
        result.append(temp)
    #print(result)
    num = 0
    for i in result:
        if i["test"][0] == i["label"]:
            num+=1
    ratio = num/n
    print("%.4f" %ratio)


def main(argv):
    if len(argv) == 1:
        print("KNN.py: A KNN algorithm program to predict the survival of a person in the titanic incident.\n")
        print("USage:\npython KNN.py filepath k [DataDivideRatio=0.7]\n")
        print("filepath\n\tthe full path of the .dat or .csv file, relative or absolute all both accepted.")
        print("k\n\tthe number of nearest neighbor, which has to be set by the user.")
    elif len(argv) == 2:
        if 48 <= ord(argv[1][0]) <= 57:
            print("Please specify the filepath.")
        else:
            print("Please specify the k number. Less than 50 is recommended.")
    elif len(argv) > 2:
        filename = argv[1]
        k = int(argv[2])
        if len(argv) == 3:
            ratio = 0.7
        else:
            ratio = float(argv[3])
        people = ReadplusStandard(filename)
        traindata,testdata = DataDevide(people,ratio)
        #label,possibility = KNNClassify(traindata,{"attribute":[0.3,1,1],"label":1},10)
        ClassTest(traindata,testdata,k)

main(argv)