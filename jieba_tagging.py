#author_by zhuxiaoliang
#2018-07-11 上午11:39
import jieba.posseg as psg
sent = '中文分词是一门艺术'
seg_list = psg.cut(sent)
print(' '.join(['{}/{}'.format(w,t) for w,t in seg_list]))
'''
中文/nz 分词/n 是/v 一门/m 艺术/n
将分词中基于字的标注和词性标注结合起来。
'''