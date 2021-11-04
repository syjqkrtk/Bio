import numpy as np

file = open("gt.txt","r")    
gt = []

for i in range(48):
    gt.append(int(file.readline().replace("\n","")))

for _ in range(10,33):
    file2 = open("mash"+str(_)+".txt","r")
    mash = []
    mash2 = []
    
    for i in range(48):
        temp = int(file2.readline().replace("\n",""))
        mash.append(2-temp)
        mash2.append(temp-1)
        
    print(max(np.sum(np.array(gt) == np.array(mash))/48,np.sum(np.array(gt) == np.array(mash2))/48))
    
    file.close()
    file2.close()