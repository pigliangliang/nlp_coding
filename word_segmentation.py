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


'''
