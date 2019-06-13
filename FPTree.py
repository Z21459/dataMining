import re
import collections
import itertools

class node:# 定义节点
    def __init__(self, val, char):
        self.val = val  # 用于定义当前的计数
        self.char = char  # 用于定义当前的字符是多少
        self.children = {}  # 用于存储孩子
        self.next = None  # 用于链表，链接到另一个孩子处
        self.father = None  # 构建条件树时向上搜索
        self.visit = 0  # 用于链表的时候观察是否已经被访问过了
        self.nodelink = collections.defaultdict()
        self.nodelink1 = collections.defaultdict()

class FPTree():
    def __init__(self):
        self.root = node(-1, 'root')
        self.FrequentItem = collections.defaultdict(int)  # 用来存储频繁项集
        self.res = []

    def BuildTree(self, data):  # 建立fp树的函数,data应该以list[list[]]的形式，其中内部的list包含了商品的名称，以字符串表示
        for line in data:  # 取出第一个list，用line来表示
            root = self.root
            for item in line:  # 对于列表中的每一项
                if item not in root.children.keys():  # 如果item不在dict中
                    root.children[item] = node(1, item)  # 创建一个新的节点
                    root.children[item].father = root  # 用于从下往上寻找
                else:
                    root.children[item].val += 1  # 否则，计数加1
                root = root.children[item]  # 往下走一步
                if item in self.root.nodelink.keys():  # 创建链表，如果这个item在nodelink中已经存在了
                    if root.visit == 0:  # 如果这个点没有被访问过
                        self.root.nodelink1[item].next = root
                        self.root.nodelink1[item] = self.root.nodelink1[item].next
                        root.visit = 1  # 被访问了

                else:  # 如果这个item在nodelink中不存在
                    self.root.nodelink[item] = root
                    self.root.nodelink1[item] = root
                    root.visit = 1
        print('树建立完成，开始查找频繁模式')
        return self.root

    def IsSinglePath(self, root):
        # print('是否为单路径')
        if not root:
            return True
        if not root.children: return True
        a = list(root.children.values())
        if len(a) > 1:
            return False
        else:
            for value in root.children.values():
                if self.IsSinglePath(value) == False: return False
            return True
    def FP_growth(self, Tree, a, HeadTable):  # Tree表示树的根节点，a用列表表示的频繁项集,HeadTable用来表示头表
        # 我们首先需要判断这个树是不是单路径的，创建一个单路径函数IsSinglePath(root)
        if self.IsSinglePath(Tree):  # 如果是单路径的
            # 对于路径中的每个组合，记作b，产生模式，b并a，support = b中节点的最小支持度
            root, temp = Tree, []  # 创建一个空列表来存储
            while root.children:
                for child in root.children.values():
                    temp.append((child.char, child.val))
                    root = child
            ans = []# 产生每个组合
            for i in range(1, len(temp) + 1):
                ans += list(itertools.combinations(temp, i))
            # print('ans = ',ans)
            for item in ans:
                mychar = [char[0] for char in item] + a
                mycount = min([count[1] for count in item])
                if mycount >= support:
                    print("构造"+str(mychar)+"-条件的FP-Tree后： "+str(mychar)+" 支持度： "+str(mycount))
                    #print()
                    self.res.append([mychar, mycount])
        else:  # 不是单路径，存在多个路径
            root = Tree
            HeadTable.reverse()  # 首先将头表逆序

            for (child, count) in HeadTable:  # child表示字符，count表示支持度
                b = [child] + a  # 新的频繁模式
                # 开始构造b的条件模式基
                print()
                print("构造"+str(b)+"-条件的FP-Tree前： "+str(b) + " 支持度： "+str(count))
                self.res.append([b, count])
                tmp = Tree.nodelink[child]  # 此时为第一个节点从这个节点开始找,tmp一直保持在链表当中
                data = []  # 用来保存条件模式基
                while tmp:  # 当tmp一直存在的时候
                    tmpup = tmp  # 准备向上走
                    res = [[], tmpup.val]  # 用来保存条件模式

                    while tmpup.father:
                        res[0].append(tmpup.char)
                        tmpup = tmpup.father

                    res[0] = res[0][::-1]  # 逆序
                    data.append(res)  # 条件模式基保存完毕
                    tmp = tmp.next
                # 条件模式基构造完毕，储存在data中，下一步是建立b的fp-Tree

                # 统计词频
                CountItem = collections.defaultdict(int)
                for [tmp, count] in data:
                    for i in tmp[:-1]:
                        CountItem[i] += count

                for i in range(len(data)):
                    data[i][0] = [char for char in data[i][0] if CountItem[char] >= support]  # 删除掉不符合的项
                    data[i][0] = sorted(data[i][0], key=lambda x: CountItem[x], reverse=True)  # 排序

                # 此时数据已经准备好了，我们需要做的就是构造条件树
                root = node(-1, 'root')  # 创建根节点，值为-1，字符为root
                for [tmp, count] in data:  # item 是 [list[],count] 的形式

                    tmproot = root  # 定位到根节点
                    for item in tmp:  # 对于tmp中的每一个商品
                        # print('123',item)
                        # CountItem1[item] += 1
                        if item in tmproot.children.keys():  # 如果这个商品已经在tmproot的孩子中了
                            tmproot.children[item].val += count  # 更新值
                        else:  # 如果这个商品没有在tmproot的孩子中
                            tmproot.children[item] = node(count, item)  # 创建一个新的节点
                            tmproot.children[item].father = tmproot  # 方便从下往上找
                        tmproot = tmproot.children[item]  # 往下走一步

                        # 根据这个root创建链表
                        if item in root.nodelink.keys():  # 这个item在nodelink中存在
                            if tmproot.visit == 0:
                                root.nodelink1[item].next = tmproot
                                root.nodelink1[item] = root.nodelink1[item].next
                                tmproot.visit = 1
                        else:  # 这个item在nodelink中不存在
                            root.nodelink[item] = tmproot
                            root.nodelink1[item] = tmproot
                            tmproot.visit = 1

                if root:  # 如果新的条件树不为空
                    NewHeadTable = sorted(CountItem.items(), key=lambda x: x[1], reverse=True)

                    for i in range(len(NewHeadTable)):
                        if NewHeadTable[i][1] < support:
                            NewHeadTable = NewHeadTable[:i]
                            break

                    self.FP_growth(root, b, NewHeadTable)  # 我们需要创建新的headtable

                # return root#成功返回条件树

    def PrintTree(self, root):  # 层次遍历打印树
        if not root: return
        res = []
        if root.children:
            for (name, child) in root.children.items():
                print(str(name)+" "+str(child.val))
                res += [name + ' ' + str(child.val), self.PrintTree(child)]
            return res
        else:
            return

if __name__ == '__main__':
    #读文件和apriori相同，简练起见，直接写数据了
    data = [['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], ['a', 'f', 'g'], ['b', 'd', 'e', 'f', 'j'], ['a', 'b', 'd', 'i', 'k'],
        ['a', 'b', 'e', 'g']]
    #data = readData("./dataOfFPTree.csv")
    print("初始数据： "+str(data))
    data = data
    support = 3
    print("最小支持度： "+str(support))
    CountItem = collections.defaultdict(int)# 统计单项的频率
    for line in data:
        for item in line:
            CountItem[item] += 1
    a = sorted(CountItem.items(), key=lambda x: x[1], reverse=True)# 将dict按照频率从大到小排序,并且删除掉频率过小的项
    for i in range(len(a)):
        if a[i][1] < support:
            a = a[:i]
            break
    for i in range(len(data)):# 更新data中，每一笔交易的商品顺序
        data[i] = [char for char in data[i] if CountItem[char] >= support]
        data[i] = sorted(data[i], key=lambda x: CountItem[x], reverse=True)
    obj = FPTree()
    root = obj.BuildTree(data)
    obj.FP_growth(root, [], a)