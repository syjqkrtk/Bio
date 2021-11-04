import os
import numpy as np
import cv2

for number in range(50):
    for number2 in range(10):

        dir1 = "D:\\REMinerII\\Ecoli\\Ecoli_"+str(number)+"_"+str(number2)+"\\Forward"
        dir2 = "D:\\REMinerII\\Ecoli\\Ecoli_"+str(number)+"_"+str(number2)+"\\Reverse"
        dir3 = "D:\\REMinerII\\Processed2"
        
        THRESHOLD = 300
        
        fname1 = os.listdir(dir1)
        fname2 = os.listdir(dir2)
        startlist = []
        endlist = []
        scorelist = []
        ACGTlist = []
        
        for name in fname1:
            fnamet = os.listdir(dir1+"\\"+name)
            for name2 in fnamet:
                fnamett = os.listdir(dir1+"\\"+name+"\\"+name2)
                for name3 in fnamett:
                    if "align" in name3:
                        info = name3.split("(")[1].split(")")[0]
                        start = info.split("_")[0]
                        end = info.split("_")[1]
                        score = int(info.split("_")[2].split(".")[2])
                        if score > THRESHOLD:
                            file = open(dir1+"\\"+name+"\\"+name2+"\\"+name3,'r')
                            data = file.read()
                            ACGT = [data.count("A"),data.count("C"),data.count("G"),data.count("T")]
                            file.close()
                            print(start, end, score)
                            startlist.append(start.split("."))
                            endlist.append(end.split("."))
                            scorelist.append(score)
                            ACGTlist.append(ACGT)
                            
            fnamet = os.listdir(dir2+"\\"+name)
            for name2 in fnamet:
                fnamett = os.listdir(dir2+"\\"+name+"\\"+name2)
                for name3 in fnamett:
                    if "align" in name3:
                        info = name3.split("(")[1].split(")")[0]
                        start = info.split("_")[0]
                        end = info.split("_")[1]
                        score = int(info.split("_")[2].split(".")[2])
                        if score > THRESHOLD:
                            file = open(dir2+"\\"+name+"\\"+name2+"\\"+name3,'r')
                            data = file.read()
                            ACGT = [data.count("A"),data.count("C"),data.count("G"),data.count("T")]
                            file.close()
                            print(start, end, score)
                            startlist.append(start.split("."))
                            endlist.append(end.split("."))
                            scorelist.append(score)
                            ACGTlist.append(ACGT)
                            
        alignstart = []
            
        for i in range(len(startlist)):                
            alignstart.append([int(startlist[i][0]), int(startlist[i][1])])
        
        alignend = []
            
        for i in range(len(startlist)):
            alignend.append([int(endlist[i][0]), int(endlist[i][1])])
                
        alignscore = []
            
        for i in range(len(startlist)):                
            alignscore.append(scorelist[i])
                
        alignACGT = []
            
        for i in range(len(startlist)):
            alignACGT.append(ACGTlist[i])
        
        file = open(dir3+"\\"+str(number)+"_"+str(number2)+".csv",'w')
        for k in range(len(alignscore)):
            xdiff = alignend[k][0] - alignstart[k][0]
            ydiff = alignend[k][1] - alignstart[k][1]
            diff = np.abs(xdiff) + np.abs(ydiff)
            file.write(str(alignstart[k][0]/6000000)+","+str(alignstart[k][1]/6000000)+","+str(alignend[k][0]/6000000)+","+str(alignend[k][1]/6000000)+","+str(2*alignscore[k]/diff)+","+str(alignACGT[k][0]/diff)+","+str(alignACGT[k][1]/diff)+","+str(alignACGT[k][2]/diff)+","+str(alignACGT[k][3]/diff)+"\n")
            
        file.close()