import numpy as np
import random
import time

# Indel length distribution을 결정하기 위한 zipfian 함수
def zipfian(s,N):
    temp0 = np.array(range(1,N+1))
    temp0 = np.sum(1/temp0**s)
    temp = random.random() * temp0

    for i in range(N):
        temp2 = 1 / ((i + 1) ** s)

        if temp < temp2:
            return i+1
        else:
            temp = temp - temp2

    return 0

# 시퀀스를 숫자 형태로 읽기 위한 함수
def ReadSeq(fname):
    file = open(fname,'r')
    SeqTemp = list(file.readline().replace("\n",""))
    Seq = np.zeros(np.size(SeqTemp))
    for _ in range(np.size(SeqTemp)):
        Seq[_] = (SeqTemp[_] == 'A') + 2 * (SeqTemp[_] == 'C') + 3 * (SeqTemp[_] == 'G') + 4 * (SeqTemp[_] == 'T') - 1
    file.close()
    
    return Seq

# 시퀀스를 주어진 변이 확률에 따라 변이시키는 함수
def Insilico(seq, snp, indel, maxI, num, count):
    l_seqs = len(seq)
    for i in range(num):
        # SNP 추가
        seqtemp = np.mod(seq + (np.random.rand(l_seqs)  < snp)*np.random.randint(4, size=l_seqs),4)
        
        # indel 발생 위치 및 정보 결정 단계
        indelonoff = (np.random.rand(l_seqs) < indel)
        indeltotal = np.sum(indelonoff)
        indelindex = np.where(indelonoff == 1)
        deleteuntil = 0
        
        indeltemp = []
        for kk in range(indeltotal):
            if indelindex[0][kk] > deleteuntil:
                indellen = zipfian(1.6,maxI)
                ranval = np.random.rand()
                if ranval < 1/2:
                    indeltemp.append(np.random.randint(4, size=indellen))
                else:
                    indeltemp.append(-indellen)
                    deleteuntil = indelindex[0][kk] + indellen
        
        # 결과 시퀀스 생성 단계
        lastindex = 0
        file = open('Hepatitis_test'+str(count)+'_'+str(i)+'.txt','a')
        for kk in range(len(indeltemp)):
            # deletion이 발생된 경우
            if np.sum(indeltemp[kk]) < 0:
                seqtemp3 = seqtemp[lastindex:indelindex[0][kk]]
                lastindex = indelindex[0][kk]-indeltemp[kk]
                seq2 = ''
                for j in range(len(seqtemp3)):
                    if seqtemp3[j] == 0:
                        seq2 = seq2 + 'A'
                    elif seqtemp3[j] == 1:
                        seq2 = seq2 + 'C'
                    elif seqtemp3[j] == 2:
                        seq2 = seq2 + 'G'
                    elif seqtemp3[j] == 3:
                        seq2 = seq2 + 'T'
                file.write(seq2)
            # insertion이 발생된 경우
            else:
                seqtemp3 = np.append(seqtemp[lastindex:indelindex[0][kk]],indeltemp[kk])
                lastindex = indelindex[0][kk]
                seq2 = ''
                for j in range(len(seqtemp3)):
                    if seqtemp3[j] == 0:
                        seq2 = seq2 + 'A'
                    elif seqtemp3[j] == 1:
                        seq2 = seq2 + 'C'
                    elif seqtemp3[j] == 2:
                        seq2 = seq2 + 'G'
                    elif seqtemp3[j] == 3:
                        seq2 = seq2 + 'T'
                file.write(seq2)
                
        # 마지막 자투리 시퀀스
        seqtemp3 = seqtemp[lastindex:l_seqs]
        seq2 = ''
        for j in range(len(seqtemp3)):
            if seqtemp3[j] == 0:
                seq2 = seq2 + 'A'
            elif seqtemp3[j] == 1:
                seq2 = seq2 + 'C'
            elif seqtemp3[j] == 2:
                seq2 = seq2 + 'G'
            elif seqtemp3[j] == 3:
                seq2 = seq2 + 'T'
        file.write(seq2)        
        file.close()

# 1번 시퀀스에 대한 Insilico sequence 생성
seq = ReadSeq('Hepatitis_1.txt')
count = 0
# SNP 생성
for i in range(1,11):
    print(count)
    Insilico(seq,0.05*i,0,10,10,count)
    count = count + 1
# indel 생성
for i in range(1,11):
    print(count)
    Insilico(seq,0,0.01*i,10,10,count)
    count = count + 1
    
# 2번 시퀀스에 대한 Insilico sequence 생성
past = time.time()
seq = ReadSeq('Hepatitis_2.txt')
now = time.time()
# SNP 생성
for i in range(1,11):
    print(count)
    Insilico(seq,0.05*i,0,10,10,count)
    count = count + 1
# indel 생성
for i in range(1,11):
    print(count)
    Insilico(seq,0,0.01*i,10,10,count)
    count = count + 1