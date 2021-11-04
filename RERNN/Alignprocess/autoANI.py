import os
import numpy as np

file0 = open("result2.txt","w")

for _ in range(49):
    for __ in range(_+1,50):
        file = open("Ecoli_"+str(_+1)+"_"+str(__+1)+".out","r")

        match = 0
        length = 0

        temp = file.readline()
        while temp:
            if "Identities" in temp:
                a = temp.split("Identities")
                b = a[1].split("/")
                c = int(b[0].split()[-1])
                d = int(b[1].split()[0])
                match += c
                length += d

            temp = file.readline()

        if length != 0:
            file0.write(str(match/length)+"\n")
        else:
            file0.write("0\n")
