import numpy as np

states = ('吃','睡')
observations = ('哭','没精神','找妈妈')
start_ = {'吃':0.3,'睡':0.7}
transition_ = {
    '吃':{'吃':0.1,'睡':0.9},
    '睡':{'吃':0.8,'睡':0.2}
}
emission_ = {
    '吃':{'哭':0.7,'没精神':0.1,'找妈妈':0.2},
    '睡':{'哭':0.3,'没精神':0.5,'找妈妈':0.2}
}

def generate_index_map(lables):
    id2label = {}
    label2id = {}
    i = 0
    for l in lables:
        id2label[i] = l
        label2id[l] = i
        i += 1
    return id2label, label2id


states_id2label, states_label2id = generate_index_map(states)
observations_id2label, observations_label2id = generate_index_map(observations)
print(states_id2label, states_label2id)
print(observations_id2label, observations_label2id)


def convert_map_to_vector(map_, label2id):#将概率向量从dict转换成一维array
    v = np.zeros(len(map_), dtype=float)
    for e in map_:
        v[label2id[e]] = map_[e]
    return v


def convertDictJuzhen(map_, label2id1, label2id2):#将概率转移矩阵从dict转换成矩阵
    m = np.zeros((len(label2id1), len(label2id2)), dtype=float)
    for line in map_:
        for col in map_[line]:
            m[label2id1[line]][label2id2[col]] = map_[line][col]
    return m

def forward(obs_seq):#前向算法
    N = A.shape[0]
    print("A"+str(A))
    T = len(obs_seq)

    F = np.zeros((N, T))# F保存前向概率矩阵
    F[:, 0] = pi * B[:, obs_seq[0]]

    for t in range(1, T):
        for n in range(N):
            F[n, t] = np.dot(F[:, t - 1], (A[:, n])) * B[n, obs_seq[t]]
    return F

def viterbi(trainsition_probability,emission_,start_,observations):
    # 最后返回一个Row*Col的矩阵结果
    Row = np.array(trainsition_probability).shape[0]
    Col = len(observations)
    #print("Row : " + str(Row)+"  Col"+str(Col))
    #定义要返回的矩阵
    F=np.zeros((Row,Col))
    #初始状态
    F[:,0]=start_*np.transpose(emission_[:,observations[0]])
    #print("F[:,0]" + str(F))
    for t in range(1,Col):
        #print("t值位："+str(t))
        list_max=[]
        for n in range(Row):
            #print("np.transpose(trainsition_probability[:,n])"+str(np.transpose(trainsition_probability[:,n])))
            list_x=list(np.array(F[:,t-1])*np.transpose(trainsition_probability[:,n]))
            #print("list_x: "+str(list_x)+"\nF: "+str(F))
            #获取最大概率
            list_p=[]
            for i in list_x:
                list_p.append(i*10000)
            list_max.append(max(list_p)/10000)
        #print("list_max"+str(list_max))
        #print("emission_[:,observations[t]]"+str(emission_[:,observations[t]]))
        F[:,t]=\
            np.array(list_max)*\
            np.transpose(emission_[:,observations[t]])
        #print("cal之后："+str(F))
    return F

A = convertDictJuzhen(transition_, states_label2id, states_label2id)
print("A:"+str(A))
B = convertDictJuzhen(emission_, states_label2id, observations_label2id)
print("B"+str(B))
observations_index = [observations_label2id[o] for o in observations]
pi = convert_map_to_vector(start_, states_label2id)
print("pi:: "+str(pi))

observations = [0,1,2]#,1,2
F=forward(observations)
print(str(F))
ans = 0.0
for indexx in F:
    ans+=indexx[-1]
print("第一问答案：序列： "+str([observations_id2label[index] for index in observations])+"  概率:  %.5f" %(ans))

F =  viterbi(A,B,pi,observations)
print("第二问答案： 矩阵\n"+str(F))
flag = 0
for yy in range(0,len(observations)):
    maxx = 0
    print("第"+str(yy+1)+"步")
    for xx in range(0,A.shape[0]):
        #print("xx,yy"+str(xx)+str(yy))
        #maxx = F[x][y]
        if maxx==0:
            maxx=F[xx][yy]
        elif F[xx][yy]>maxx :
            maxx=F[xx][yy]
            flag=xx
            #print("ssssssssssssss")
            #print(states_id2label)
            print(states_id2label[1])

        elif xx==A.shape[0]-1 and maxx==F[xx][yy]:
            for statuss in states_id2label.keys():
                print(states_id2label[statuss],end='')
                if statuss!=len(states_id2label)-1:
                    print("或",end='')
            print()
        else:
            print(states_id2label[0])