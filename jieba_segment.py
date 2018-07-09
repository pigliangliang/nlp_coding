#author_by zhuxiaoliang
#2018-07-09 下午7:12
"""


import jieba
sent = '在伸手不见五指的夜晚，我看见了一只小野猫从远处跑过去。'''
seg_list = jieba.cut(sent,cut_all=True)
print('全模式：',' '.join(seg_list))

seg_list = jieba.cut(sent,cut_all=False)
print('精确模式：',' '.join(seg_list))

seg_list = jieba.cut_for_search(sent)
print('搜索模式：',' '.join(seg_list))
"""
'''
全模式：   在 伸手 伸手不见 伸手不见五指 不见 五指 的 夜晚   我 看见 了 一只 小野 野猫 从 远处 跑 过去  
精确模式： 在 伸手不见五指 的 夜晚 ， 我 看见 了 一只 小 野猫 从 远处 跑 过去 。
搜索模式： 在 伸手 不见 五指 伸手不见五指 的 夜晚 ， 我 看见 了 一只 小 野猫 从 远处 跑 过去 。

'''
#高频词汇提取

#数据读取
def get_content(path):
    with open(path,'r',encoding='utf-8',errors='ignore') as f:
        content = ''
        for l in f:
            l = l.strip()
            content +=l
        return l
#统计高频词汇
def get_tf(words,topk=10):
    tf_dic = {}
    for w in words:
        tf_dic[w]=tf_dic.get(w,0)+1
    return sorted(tf_dic.items(),key=lambda x:x[1],reverse=True)[:topk]
#使用结巴分词
def main():
    import jieba

    content  = get_content('jieba')
    seg_list = jieba.cut(content)
    #print('分词结果：',' '.join(seg_list))

    ret = get_tf([ i for i in seg_list])
    print(ret)
main()


ls = []
with open('hmm','r') as f:#语料库是词语用空格隔开，直接分词
    for l in f.readlines():
        ls.extend([i for i in l[:-1].split() if i !=' ' and i !='\n' and i !='，' and i != '。' \
                   and i != '、'])
from collections import Counter
d = Counter(ls)
#print(sorted(Counter(ls).items(),key=lambda x:x[1],reverse=True)[:10])
'''
[('的', 173), ('在', 48), ('和', 46), ('是', 34), ('发展', 32), ('了', 27), ('关系', 26), ('我们', 25), ('说', 25), ('中', 24)]
'''