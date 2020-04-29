"""
这个文件用于使用16部金庸小说训练LDA模型
"""
import jieba
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import torch
import pandas as pd
import argparse

# 参数设置
parser = argparse.ArgumentParser(description='Propert parameters for lda text classification')
parser.add_argument('--k', default=10, type=int, metavar='K',
                    help='num of topic')
parser.add_argument('--test', default='test.json', type=str, metavar='test_path',
                    help='path to test')
args = parser.parse_args()
# 准备数据
jieba.load_userdict("人名专有词词典.txt")
with open("stopwords.txt", "r", encoding="utf-8") as f:
    stopwords = [eve.strip("\n") for eve in f]
with open('训练集/train.json', 'r') as f:
    corpus = json.load(f)
with open('测试集/'+args.test, 'r') as f:
    test = json.load(f)
print("读取语料库中数据集：", len(corpus)+len(test))
data = []
labels = []
for key in corpus.keys():
    data.append(" ".join(corpus[key]))
    labels.append(key)
assert len(data) == 16
for key in test.keys():
    data.append(test[key])
    labels.append(key.split("_")[0])
# print(data[0][:5])

# data = data[16:]
# labels = labels[16:]

# 计算文档的词频向量
fileVector = CountVectorizer(stop_words=stopwords)

fileTfVector = fileVector.fit_transform(data)

print(fileTfVector.shape)

# LDA训练  使用16部小说进行训练
topic = args.k
model = LDA(n_components=topic, max_iter=50, learning_method='batch')
docres = model.fit_transform(fileTfVector[:16])

# print(docres)
value, indices = torch.max(torch.tensor(docres), 1)
print(indices)
print("{}个主题识别出来了{}个主题".format(topic, len(list(set(indices.tolist())))))
# print(len(model.components_))

res = model.transform(fileTfVector)
assert len(res) == len(labels)
df_labels = pd.DataFrame(labels)
# df_labels.to_excel("labels.xlsx")
df_res = pd.DataFrame(res)
# df_res.to_excel("ldaVector.xlsx")
df = pd.concat([df_labels, df_res], axis=1)
df.to_excel("labels_with_vector.xlsx")


with open("history.txt", "a", encoding="utf-8") as f:
    print(topic, file=f)
    print(model.get_params(), file=f)
    print("perplexity:", model.perplexity(fileTfVector[:16]), file=f)
    print("", file=f)
