file = open("name.txt","r")
file2 = open("name2.txt","w")
temp = True
while temp:
    temp = file.readline()
    file2.write(temp.replace("\n","").replace(" ","").replace("'","").replace('"',"")+"\n")
    
file.close()
file2.close()