for i in range(50):
    file = open("Ecoli_"+str(i+1)+".txt","r")
    file2 = open("Ecoli_"+str(i+1)+".fna","w")

    txt = file.read()
    file2.write(">Ecoli_"+str(i+1)+"\n")
    file2.write(txt)
    file2.close()
    file.close()
