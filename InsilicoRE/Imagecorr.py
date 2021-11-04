import cv2

# 1번 시퀀스에 대한 유사도 측정
file = open('result_1.txt','w')
path2 = 'Processed\\Processed_Original_1.png'
template = cv2.imread(path2)
for i in range(400):
    path1 = 'Processed\\Processed_'+str(i+1)+'.png'
    img = cv2.imread(path1)
    img = img
    template = template
    method = cv2.TM_CCOEFF_NORMED
    result = cv2.matchTemplate(img,template,method)
    print (i, result[0][0])
    file.write(str(result[0][0])+'\t')
    if (i+1)%10 == 0:
        file.write('\n')
file.close()

# 2번 시퀀스에 대한 유사도 측정
file = open('result_2.txt','w')
path2 = 'Processed\\Processed_Original_2.png'
template = cv2.imread(path2)
for i in range(400):
    path1 = 'Processed\\Processed_'+str(i+1)+'.png'
    img = cv2.imread(path1)
    img = img
    template = template
    method = cv2.TM_CCOEFF_NORMED
    result = cv2.matchTemplate(img,template,method)
    print (i, result[0][0])
    file.write(str(result[0][0])+'\t')
    if (i+1)%10 == 0:
        file.write('\n')
file.close()