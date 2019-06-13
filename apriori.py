import copy

def binaryfindsub(items): #二进制枚举找找子集
    N = len(items)
    for i in range(2**N):
        combo = []
        for j in range(N):
            if(i >> j ) % 2 == 1:
                combo.append(items[j])
        yield combo

def readData(fileName):
    f=open(fileName,'r',encoding="utf-8")
    lines=f.readlines()
    dataSet=[]
    for line in lines[0:]:
        print("line: "+line)
        line=line.strip().split(',')
        dataSet.append(line)
    print(str(dataSet))
    return dataSet

def createcandidates(dataSet):#构建初始单项集
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    return [frozenset(var) for var in C1]


def scanDataSet(D, Ck, minSupport):#扫描数据库，统计支持度
    subSetCount = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):# 检查候选k项集中的每一项的所有元素是否都出现在每一个事务中，若true，则加1
                subSetCount[can] = subSetCount.get(can, 0) + 1# subSetCount为候选支持度计数，get()返回值，如果值不在字典中则返回默认值0。
    numItems = float(len(D))
    returnList = []
    supportData = {}# 选择出来的频繁项集，未使用先验性质
    for key in subSetCount:
        support = subSetCount[key] / numItems  # 每个项集的支持度
        if support >= minSupport:  # 将满足最小支持度的项集，加入returnList
            print("Lk: k=" + str(len(key)) + "  " + str(set(key)) + " support： " + str(support))
            returnList.insert(0, (key))
            supportData[key] = support  # 汇总支持度数据
    return returnList, supportData


def aprioriGen(Lk, k):  # Aprior算法生成候选Ck
    Ck = []
    for i in range(len(Lk)):
        L1 = list(Lk[i])[: k - 2]# 只需取前k-2个元素相等的候选频繁项集即可组成元素个数为k+1的候选频繁项集
        for j in range(i + 1, len(Lk)):
            L2 = list(Lk[j])[: k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:#集合的union方法
                Ck.append((Lk[i]) | (Lk[j]))
    return Ck

def has_infrequent_subset(L, Ck, k):#剪枝，任一频繁项集的所有非空子集也必须是频繁的，反之，如果某个候选的非空子集不是频繁的
    Ckc = copy.deepcopy(Ck)#复制
    for i in Ck:
        p = [t for t in i]
        i_subset = binaryfindsub(p)
        subsets = [i for i in i_subset]
        for each in subsets:
            if each!=[] and each!=p and len(each)<k:
                if frozenset(each) not in [t for z in L for t in z]:
                    Ckc.remove(i)
                    break
    return Ckc


def apriori(dataSet, minSupport):#apriori算法主框架函数
    print("统计每个单项的支持度：")
    C1 = createcandidates(dataSet)# 构建初始候选项集C1
    D = [set(var) for var in dataSet]#形成集合
    L1, LS = scanDataSet(D, C1, minSupport)# 构建初始的频繁项集
    L = [L1]# L初始为最初的L1
    k = 2# 项集应该含有2个元素，所以 k=2
    while (len(L[k - 2]) > 0):
        print("开始查找频繁"+str(k)+"项集：")
        Ck = aprioriGen(L[k - 2], k)
        print("Ck: "+str(Ck))
        Ck2 = has_infrequent_subset(L, Ck, k)# 剪枝，减少计算量
        Lk, supK = scanDataSet(D, Ck2, minSupport)# 候选项集支持度和最小支持度进行比较,得到Lk
        LS.update(supK)# 将新的项集的支持度数据加入原来的结果字典中
        L.append(Lk)#添加生成的频繁项集
        k += 1#k结束，k自增
    return L[:-1], LS# 返回所有满足条件的频繁项集的列表，和所有候选项集的支持度信息


if __name__ == '__main__':
    myDat = readData("./dataOfApriori.csv")
    print("初始数据:  "+str(myDat))
    L, LS = apriori(myDat, 0.4)