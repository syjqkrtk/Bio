import numpy as np

file = open("gt.txt","r")
file2 = open("blast.txt","r")
file3 = open("mash.txt","r")

gt = []
blast = []
mash = []

for i in range(48):
    gt.append(int(file.readline().replace("\n","")))
    blast.append(2-int(file2.readline().replace("\n","")))
    mash.append(2-int(file3.readline().replace("\n","")))
    
print(np.sum(np.array(gt) == np.array(blast))/50)
print(np.sum(np.array(gt) == np.array(mash))/50)
print(np.sum(np.array(mash) == np.array(blast))/50)

file.close()
file2.close()
file3.close()