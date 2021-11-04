import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree

TAG = [1,1,1,1,1,0,0,0,0,0,0,1,0,0,1,0,1,0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,0,1]

class data():
    def __init__(self, tag, num, snp):
        self.len = 0
        self.data = []
        self.dist = []
        self.Z = []
        self.label = []
        self.tag = tag
        self.num = num
        self.file = open("Text/SNP/RE_"+str(snp)+".txt","a")
        self.file2 = open("Text/SNP/embed_word_"+str(snp)+".csv","a")

    def append(self, v):
        temp = []
        for i in v:
            temp.append(float(i))
        if float(v[4])>0:
            self.label.append(0)
        else:
            self.label.append(1)
        self.data.append(temp)
        self.len += 1

    def calcdist(self):
        self.dist = []
        for i in range(self.len-1):
            for j in range(i+1,self.len):
                self.dist.append(distance.euclidean(self.data[i], self.data[j]))

    def clustering(self):
        self.Z = linkage(self.dist,"average")

    def visualize(self, name = 0):
        plt.figure(figsize=(10, 4))

        ddata = dendrogram(self.Z)

        dcoord = np.array(ddata["dcoord"])
        icoord = np.array(ddata["icoord"])
        idx = np.argsort(dcoord[:, 2])
        dcoord = dcoord[idx, :]
        icoord = icoord[idx, :]

        if name:
            plt.savefig(name)
            plt.close()
        else:
            plt.show()
            plt.close()

    def embed_word(self):
        for i in range(self.len):
            self.file2.write(str(10000*self.num+self.nodelist[i].id)+",")
            for j in range(9):
                self.file2.write(str(self.data[i][j])+",")
            self.file2.write("\n")
        self.file2.close()

    def print(self):
        self.rootnode, self.nodelist = to_tree(self.Z, rd=True)
        self.text = ""
        self.printAEBTR(self.rootnode.id)
        self.text = "("+str(self.tag)+self.text[2:]
        self.file.write(self.text+"\n")
        self.file.close()

    def printAEBTR(self, currNode):
        if currNode is None:
            return
        left = self.nodelist[currNode].left
        right = self.nodelist[currNode].right

        if left is not None:
            self.text += "(0 "
            self.printAEBTR(left.id)
        if self.nodelist[currNode].count == 1:
            self.text += "(0 "+str(10000*self.num+self.nodelist[currNode].id)+")"
        else:
            self.text += " "
        if right is not None:
            self.printAEBTR(right.id)
            self.text += ")"

for snp in [0.0001,0.0002,0.0003,0.0005,0.001,0.002,0.003,0.005,0.01,0.02,0.03,0.05,0.1]:
    for i in range(50):
        if i == 35:
            continue

        for j in range(20):
            print(snp, i,j)
            tempd = data(3*TAG[i], i*100+j, snp)
            filename = "Processed2/"+str(snp)+"/Ecoli_"+str(i)+"_"+str(j)+".csv"
            file = open(filename,"r")
            while True:
                temp = file.readline().replace("\n","").split(",")
                if np.size(temp) <= 1:
                    break
                else:
                    tempd.append(temp)

            tempd.calcdist()
            tempd.clustering()
            tempd.print()
            tempd.embed_word()
            file.close()
