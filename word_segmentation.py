#author_by zhuxiaoliang
#2018-07-08 下午7:05

"""
一、分词技术

1 规则分词
   1）正向最大分词
        基本思想是：假定分词词典中最长词有一个汉字字符，则用被处理文档的当前字串的前i个字作为匹配字段，查找
        字典，若字典中存在这样的一个长度为i的字词，则匹配成功，匹配字段被做为一个词切分出来。如果字典中
        没有找到这个词，则匹配失败，将词的最后一个字去掉，对剩下的字串重新进行匹配处理。
        如此进行，知道匹配成功，即切出一个词。

   2）逆向最大匹配
        思想与正向相反。
   3）双向最大匹配
        思想：将正向与反向最大分词的结果进行比较，按照最大匹配原则，选取词数切分最少的最为结果。


    示例：正向最大匹配代码
特点：简单，准确度高，但是必须维护字典，不能覆盖所有词。

#class IMM(object):
    def __init__(self,dic_path):
        self.dictionary = set()
        self.maximum = 0
        with open(dic_path,'r',) as f:
            for x in (f.read().split()):
                self.dictionary.add(x)
                if len(x)>self.maximum:
                    self.maximum = len(x)
    def cut(self,text):
        result = []
        index = len(text)
        while index:
            word = None
            if index-self.maximum<=0:
                piece=text[:index]
                if piece in self.dictionary:
                    word = piece
                    result.append(text[:index])
                    text = text[index:]
                    index = len(text)

            if word is None:
                index -=1
#        return result

if __name__ == '__main__':
    text = '我爱南京市长江大桥你呢'#目前不支持文本有空格，符号等
    tokenizer = IMM('imm_dic')
    print(tokenizer.dictionary)
    print(tokenizer.maximum)
    print(tokenizer.cut(text))
'''
输出：
{'南京市', '你', '大桥', '南京', '长江大桥', '呢', '玛逼', '曹尼', '爱', '长江', '我'}
4
['我', '爱', '南京市', '长江大桥', '你', '呢']
'''"""

'''
2、统计分词
主要思想是：
把每个词看作是由词的最小单位的各个字组成的，如果相连的字在不同的文本中出现次数越多，就证明这相连的字就很可能就是一个词。
因此我们就可以利用字与字相邻出现的频率来反应词的可靠性，统计语料中相邻共现的各个字组合的频率，当这个频率高于某个临界值，
就可以组成一个词。
基于统计的分词：
1）建立统计语言模型
2）对句子进行单词划分，然后对分词结果进行概率计算，获得概率最大的分词结果。比如：HMM 和CRF

语言模型：
用概率论的专业术解释语言模型就是：为长度为m的字符串确定其概率分布。p(w1,w2.....wm),其中w1...wm依次表示文本中的各个词语。一般用链式法则计
算概率值 p(w1,w2....)= P(w1)P(w2|w1)P(w3|w1,w2)....P(wm|w1,w2....wm-1)
上式当文本过长，计算难度越来越大。
n——gram模型：
n=1:p(w1,w2....)=p(w1)p(w2)...p(wm)
n=2:p(w1,w2....)=p(w1)p(w2|pw1)...p(wi|wi-1)
n=2:p(w1,w2....)=p(w1)p(w2|pw1)...p(wi|wi-1,wi-2)

HMM模型：
隐马尔科夫模型将分词作为字在字符串中序列标注任务来实现的。基本思路是：每个字在构造一个特定的词语时都占据着一个确定的构词位置，规定每个字最多有
四个构词位置：B词首，M词中，E词尾，S单独成词。
用数学抽象表示：
X=X1X2...Xn代表输入的句子，n表示句子的长度，Xn表示句中的字，Y=Y1Y2...Yn，为BMSE四中标记
这里引入观测独立性行假设：即每个字段输出仅仅与当前字有关
于是能得到公式：p(y1y2...yn|x1x2...xn) = p(y1|x1).....p(yn|xn)
基于观测独立性假设，完全没有考虑上下文，计算容易，但是会出现不合理情况。
引入HMM解决这个问题。
上式中解决p（y|x）问题，根据贝叶斯公式，p（Y|X）= p(x|y)p(y)/p（x）
对p（x|y）做马尔科夫驾驶
p（x|y）=p(x1|y1)p(x2|y2)...p(xn|yn)
对p(y)引入马尔科夫另一假设，每个输出仅仅个上一个输出有关系。
p（y）= p(y1)p(y2|y1)...p(yn|yn-1)
求解HMM的过程中使用了vetervi算法：核心思想是如果最终的最优路径经过某个xi，那么从最初节点到xi-1节点的路径也是最优路径。
python 实现HMM


'''
class HMM(object):
    def __init__(self):
        import os
        self.model_file = 'data/hmm_model.pkl'
        self.state_list = ['B','M','E','S']
        self.load_para = False
    def try_load_model(self,trained):
        if trained:
            import pickle
            with open(self.model_file,'rb') as f :
                self.A_dic = pickle.load(f)
                self.B_dic = pickle.load(f)
                self.Pi_dic = pickle.load(f)
                self.load_para = True
        else:
            #转移概率(状态到状态条件概率)
            self.A_dic = {}
            #发射概率(状态到词语概率)
            self.B_dic = {}
            #状态的初始概率
            self.Pi_dic = {}
            self.load_para = False

    def train(self,path):
        #充值概率矩阵
        self.try_load_model(False)
        #统计状态出现次数
        Count_dic = {}
        #初始化参数
        def init_parameters():
            for state in self.state_list:
                self.A_dic[state]={s:0.0 for s in self.state_list}
                self.Pi_dic[state]=0.0
                self.B_dic[state]={}
                Count_dic[state]=0
        def maklabel(text):
            cut_text = []
            if len(text) == 1:
                cut_text.append("S")
            else:
                cut_text += ['B'] +['M']*(len(text)-2)+['E']
            return cut_text
        init_parameters()
        line_num = -1
        words = set()
        with open(path,encoding="utf-8") as f:
            for line in f:
               # print(line)
                line_num +=1
                if not  line:
                    continue
                word_list = [i for i in line if i !=' ' and i !='\n']
                words |=set(word_list)
                linelist = line.split()
                line_state = []
                for w in linelist:
                    line_state.extend(maklabel(w))
                assert len(word_list)==len(line_state)
                for k,v in enumerate(line_state):
                    Count_dic[v] += 1
                    if k == 0:
                        #每个句子的第一个字的状态，用于计算初始状态概率
                        self.Pi_dic[v] +=1
                    else:
                        self.A_dic[line_state[k-1]][v] +=1
                        self.B_dic[line_state[k]][word_list[k]] = \
                            self.B_dic[line_state[k]].get(word_list[k],0) +1.0
                #print(Count_dic.items())
                #print(self.Pi_dic.items())
                #print(self.B_dic.items())
                #print(self.A_dic.items())
        self.Pi_dic = {k:v*1.0 / line_num for k,v in self.Pi_dic.items()}
        self.A_dic = {k:{k1:v1/Count_dic[k] for k1 ,v1 in v.items()} for k,v in self.A_dic.items()}
        self.B_dic = {k: {k1: (v1 + 1) / Count_dic[k] for k1, v1 in v.items()} for k,v in self.B_dic.items()}
        #print(self.B_dic.items())
        #print(self.A_dic.items())
        #print(self.Pi_dic.items())
        import pickle
        with open(self.model_file,'wb') as f:
            pickle.dump(self.A_dic,f)
            pickle.dump(self.B_dic,f)
            pickle.dump(self.Pi_dic,f)
        return self
    #基于动态规划算法，求最该概率的路径问题。
    def veterbi(self,text,states,start_p,trans_p,emit_p):
        V = [{}]
        path = {}
        for y in  states:
            V[0][y] = start_p[y]*emit_p[y].get(text[0],0)
            path[y] =[y]
        for t in range(1,len(text)):
            V.append({})
            newpath = {}

            neverSeen = text[t] not in emit_p['S'].keys() and \
            text[t] not in emit_p['M'].keys() and \
            text[t] not in emit_p['E'].keys() and \
            text[t] not in emit_p['B'].keys()

            for y in states:
                emitP =emit_p[y].get(text[t],0) if not  neverSeen else 1.0

                (prob,state) = max([(V[t-1][y0]*trans_p[y0].get(y,0) *emitP,y0) for y0 in states if V[t-1][y0]>0])
                V[t][y] = prob
                newpath[y] = path[state]+[y]
            path = newpath

        if emit_p['M'].get(text[-1],0)>emit_p['S'].get(text[-1],0):
            (prob,state)=max([(V[len(text)-1][y],y) for y in ('E','M')])
        else:
            (prob, state) = max([(V[len(text) - 1][y], y) for y in states])
        print(prob,path[state])
        return (prob,path[state])
    def cut(self,text):
        import os
        if not self.load_para:
            self.try_load_model(os.path.exists(self.model_file))
        prob,pos_list = self.veterbi(text,self.state_list,self.Pi_dic,self.A_dic,self.B_dic)
        begin,next = 0,0
        for i ,char in enumerate(text):
            pos = pos_list[i]
            if pos == 'B':
                begin = i
            elif pos == 'E':
                yield text[begin:i+1]
            elif pos == 'S':
                yield char
                next = i +1
        if next <len(text):
            yield text[next:]

hmm = HMM()

hmm.train('data/hmm')

for h in hmm.cut('在这个美丽的夜晚我睡着了'):
    print(h,end= ' ')
"""
在 这个 美丽 的 夜晚 我睡 着 了 
"""


#初始状态概率：
# dict_items([('B', 0.6612903225806451), ('M', 0.0), ('E', 0.0), ('S', 0.3419354838709677)])
#意义：作为第一个字的概率，BS可以作为文本的第一个字，B的概率大一些。而ME是不能作为首个字的，概率为零，可以看出模型输出基本正确。

#发射概率:值得是状态到字的一个概率映射。
#部分输出：

#dict_items([('B', {'中': 0.010480887792848335, '儿': 0.0012330456226880395, '踏': 0.0012330456226880395, '全': 0.007398273736128237, '各': 0.010480887792848335, '人': 0.012946979038224414, '共': 0.004932182490752158, '领': 0.010480887792848335, '建': 0.006781750924784217, '特': 0.0012330456226880395, '社': 0.012946979038224414, '道': 0.002466091245376079,
#  '改': 0.009864364981504316, '开': 0.007398273736128237, '奋': 0.0012330456226880395, '胜': 0.0012330456226880395, '前': 0.0036991368680641184, '经': 0.009864364981504316, '体': 0.0036991368680641184, '纵': 0.0012330456226880395, '稳': 0.008014796547472256, '发': 0.02527743526510481, '迈': 0.0012330456226880395, '步': 0.0018495684340320592, '生': 0.0036991368680641184, '其': 0.0018495684340320592, '事': 0.0036991368680641184, '完': 0.0030826140567200987, '七': 0.0012330456226880395, '计': 0.0012330456226880395, '第': 0.004315659679408138, '任': 0.0030826140567200987, '继': 0.007398273736128237, '有': 0.004932182490752158, '安': 0.0036991368680641184, '党': 0.0036991368680641184, '风': 0.0012330456226880395, '工': 0.0036991368680641184, '取': 0.0036991368680641184, '进': 0.012946979038224414, '民': 0.010480887792848335, '法': 0.0030826140567200987, '不': 0.006165228113440197, '加': 0.006165228113440197, '十': 0.004315659679408138, '六': 0.0018495684340320592, '通': 0.0012330456226880395, '关': 0.01911220715166461, '精': 0.0018495684340320592, '指': 0.004932182490752158, '方': 0.006781750924784217, '决': 0.0030826140567200987, '文': 0.005548705302096177, '正': 0.006781750924784217, '水': 0.0012330456226880395, '三': 0.0012330456226880395, '实': 0.007398273736128237, '伟': 0.0012330456226880395, '历': 0.004315659679408138, '转': 0.0012330456226880395, '现': 0.004315659679408138, '政': 0.012330456226880395, '持': 0.0030826140567200987, '协': 0.004315659679408138, '已': 0.0018495684340320592, '以': 0.0030826140567200987, '时': 0.004932182490752158, '之': 0.0030826140567200987, '这': 0.004315659679408138, '一': 0.011097410604192354, '大': 0.004932182490752158, '局': 0.004315659679408138, '容': 0.0012330456226880395, '因': 0.0036991368680641184, '我': 0.009864364981504316, '制': 0.004932182490752158, '系': 0.0012330456226880395, '路': 0.0012330456226880395, '符': 0.0012330456226880395, '利': 0.0036991368680641184, '愿': 0.0018495684340320592, '客': 0.0012330456226880395, '规': 0.0012330456226880395, '要': 0.002466091245376079, '根': 0.004315659679408138, '团': 0.0018495684340320592, '马': 0.002466091245376079, '毛': 0.002466091245376079, '专': 0.002466091245376079, '基': 0.009864364981504316, '原': 0.009247842170160296, '立': 0.0030826140567200987, '革': 0.0012330456226880395, '必': 0.002466091245376079, '结': 0.0018495684340320592, '没': 0.004932182490752158, '如': 0.0018495684340320592, '还': 0.0030826140567200987, '为': 0.002466091245376079, '放': 0.002466091245376079, '削': 0.0012330456226880395, '引': 0.0012330456226880395, '外': 0.005548705302096177, '先': 0.0018495684340320592, '技': 0.002466091245376079, '管': 0.0012330456226880395, '研': 0.0012330456226880395, '当': 0.002466091245376079, '思': 0.0036991368680641184, '批': 0.0012330456226880395, '吸': 0.0012330456226880395, '概': 0.0012330456226880395, '科': 0.0030826140567200987, '最': 0.0018495684340320592, '成': 0.005548705302096177, '意': 0.0036991368680641184, '可': 0.0018495684340320592, '今': 0.0030826140567200987, '需': 0.0012330456226880395, '资': 0.008631319358816275, '个': 0.0012330456226880395, '值': 0.0012330456226880395, '注': 0.0012330456226880395, '阶': 0.002466091245376079, '虽': 0.0018495684340320592, '消': 0.004315659679408138, '范': 0.0012330456226880395, '长': 0.0030826140567200987, '存': 0.002466091245376079, '忘': 0.0012330456226880395, '少': 0.0012330456226880395, '敌': 0.0018495684340320592,
#可以看到输出中是某个状态下的这些单个字的概率，意思是比如中，可以作为B的概率。可以看出中作为词首的概率很大。

#状态转移概率
#dict_items([('B', {'B': 0.0, 'M': 0.12453760789149199, 'E': 0.8754623921085081, 'S': 0.0}), ('M', {'B': 0.0, 'M': 0.3746130030959752, 'E': 0.6253869969040248, 'S': 0.0}), ('E', {'B': 0.5191122071516646, 'M': 0.0, 'E': 0.0, 'S': 0.4808877928483354}), ('S', {'B': 0.44608223429014743, 'M': 0.0, 'E': 0.0, 'S': 0.3126454615981381})])
#从上可以看出 B状态，也就是词首后面再接词首的概率基本为零，这里输出是正确的，比如M词中，接词首B的概率也是为零。