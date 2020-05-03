import numpy as np
from sklearn.ensemble import RandomForestClassifier

domainList = []
domainFeatureList = []
labelList = []

def initialize_data(file):
    f = open(file)
    for line in f:
        line = line.strip()
        if line == "":
            continue
        token = line.split(",")
        domain = token[0]
        label = token[1]
        domainList.append(Domain(domain, label))
    f.close()


def numbers(s):
    um = 0
    for l in s:
        if l.isdigit():
            um += 1
    return um


def entropy(s):
    letter_list = set(s)
    en = 0.0
    for letter in letter_list:
        p = s.count(letter) / len(s)
        en -= p * np.log(p)
    return en


class Domain:
    def __init__(self, domain, label):
        self.domain = domain
        self.label = label
        self.domainLength = len(self.domain)
        self.domainNumbers = numbers(self.domain)
        self.domainEntropy = entropy(self.domain)

    def returnInfo(self):
        return [self.domainLength, self.domainNumbers, self.domainEntropy]

    def returnLable(self):
        if self.label == "dga":
            return 1
        else:
            return 0


if __name__ == '__main__':
    initialize_data("train.txt")
    for Domain in domainList:
        domainFeatureList.append(Domain.returnInfo())
        labelList.append(Domain.returnLable())

    clf = RandomForestClassifier(random_state=0)
    clf.fit(domainFeatureList, labelList)

    testDomain = []
    testData = []
    f = open("test.txt")
    for line in f:
        line = line.strip()
        if line == "":
            continue
        testDomain.append(line)
        testData.append([len(line), numbers(line), entropy(line)])
        testLabels = clf.predict(testData)
        output = list(zip(testDomain, testLabels))

    f = open("result.txt", "w+")
    for domain, label in output:
        line = domain
        if label == 0:
            line = line + ",notdga\n"
        else:
            line = line + ",dga\n"
        f.write(line)
    f.close()
