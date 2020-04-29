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
from sklearn import metrics
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import fowlkes_mallows_score

# 参数设置
parser = argparse.ArgumentParser(description='Propert parameters for lda text classification')
parser.add_argument('--k', default=10, type=int, metavar='K',
                    help='num of topic')
parser.add_argument('--test', default='测试集/test7500.json', type=str, metavar='test_path',
                    help='path to test')
parser.add_argument('--ischar', default=False, type=bool, metavar='test_path',
                    help='using word or char')
args = parser.parse_args()
# 准备数据
jieba.load_userdict("人名专有词词典.txt")
with open("stopwords.txt", "r", encoding="utf-8") as f:
    stopwords = [eve.strip("\n") for eve in f]
with open('训练集/train.json', 'r') as f:
    corpus = json.load(f)
with open(args.test, 'r') as f:
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

print(type(data[0]))

# data = data[16:]
# labels = labels[16:]

# 计算文档的词频向量
if args.ischar:
    # 按照单个词
    import copy
    data = ["".join(eve.split()) for eve in data]  # 去除空格
    # data_ = copy.deepcopy(data)
    # for i in range(len(data_)):
    #     for j in range(len(data_[i])):
    #         if data_[i][j] in stopwords:
    #             data[i].remove(data_[i][j])
    # data = ["".join(eve) for eve in data]
    fileVector = CountVectorizer(analyzer="char")
    print("char counter is finished!")
else:
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
    print("测试数据集：", args.test, file=f)
    print(model.get_params(), file=f)
    print("perplexity:", model.perplexity(fileTfVector[:16]), file=f)
    print("", file=f)

"""
根据已经得到的lda向量进行分类
分类方法：
1-直接计算其与各个类别的余弦相似度
2-利用真实标签传入神经网络进行训练
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

df = pd.read_excel("labels_with_vector.xlsx").iloc[:, 1:]
data_df = df.iloc[:, 1:]
label_df = df.iloc[:, :1]

data = np.array(data_df).tolist()
label = np.array(label_df).tolist()

# 将训练数据与测试数据分离
split_point = 16
train_data = data[:split_point]
test_data = data[split_point:]
train_label = label[:split_point]
test_label = label[split_point:]

# 对标签进行转换
label_map = dict()
for i in range(split_point):
    label_map[train_label[i][0]] = i
true = [label_map[eve[0]] for eve in test_label]

# 计算余弦相似度
similar_matrix = cosine_similarity(test_data, train_data)
_, predict = torch.max(torch.tensor(similar_matrix), 1)
predict = predict.tolist()
print(predict)
print(true)
print("准确度：", accuracy_score(predict, true))
print("精确率：", precision_score(predict, true, average='weighted'))
print("召回率：", recall_score(predict, true, average='weighted'))
print("F1 score：", f1_score(predict, true, average='weighted'))

# 存入texmp.txt中作为运行记录保存
with open("history.txt", "a", encoding="utf-8") as f:
    print(predict, file=f)
    print(true, file=f)
    print("准确度：", accuracy_score(predict, true), file=f)
    print("精确率：", precision_score(predict, true, average='weighted'), file=f)
    print("召回率：", recall_score(predict, true, average='weighted'), file=f)
    print("F1 score：", f1_score(predict, true, average='weighted'), file=f)
    print("ami:", adjusted_mutual_info_score(predict, true), file=f)
    print("v_measure:", metrics.v_measure_score(predict, true), file=f)
    print("fmi:", fowlkes_mallows_score(predict, true), file=f)
    print("ari:", metrics.adjusted_rand_score(predict, true), file=f)

    print("\n", file=f)

with open("forplotdata.txt", "a", encoding="utf-8") as f:
    print("[",precision_score(predict, true, average='weighted'), end=",",file=f)
    print(recall_score(predict, true, average='weighted'), end=",",file=f)
    print(f1_score(predict, true, average='weighted'), end=",",file=f)
    print(metrics.adjusted_rand_score(predict, true), end="],",file=f)
    print("\n", file=f)
