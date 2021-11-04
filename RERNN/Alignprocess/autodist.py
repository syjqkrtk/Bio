file = open("result2.txt","w")

for _ in range(49):
    for __ in range(_+1,50):
        file2 = open("Ecoli_"+str(_+1)+"_"+str(__+1)+".out","r")
        temp = file2.readline()
        if temp:
            data = float(temp.split()[2])
            file.write(str(data)+"\n")
        else:
            file.write("0\n")

file.close()
