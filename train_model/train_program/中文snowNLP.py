from snownlp import SnowNLP
word = u'计算速度特别快'
# snownlp用于中文分词，类似于textblob
s = SnowNLP(word)
print(s.words)
print(list(s.tags))
print(s.sentiments)  # 可以直接得出情感极性

# 关键词的提取
text = u'''
    自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。
    它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。
    自然语言处理是一门融语言学、计算机科学、数学于一体的科学。
'''
s = SnowNLP(text)
print(s.keywords(limit=3))  # 提取三个关键词

# 提取文本摘要,
summary = s.summary(limit=2)
for i in summary:
    print(i)
