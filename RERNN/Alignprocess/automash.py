import os
import time

file = open("result.txt","w")

for _ in range(50):
    for __ in range(_+1,50):
        print(_,__)
        now = time.time()
        os.system("mash dist Ecoli_"+str(_+1)+".fna Ecoli_"+str(__+1)+".fna > Ecoli_"+str(_+1)+"_"+str(__+1)+".out")
        file.write(str(_+1)+","+str(__+1)+","+str(time.time()-now)+"\n")

file.close()
