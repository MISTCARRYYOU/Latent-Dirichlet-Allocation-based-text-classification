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
    print("\n", file=f)
