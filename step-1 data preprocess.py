"""
预处理后的小说保存在 预处理后的小说全集 目录下
"""
import os
import json
from collections import Counter
import re
import jieba
import random

def DFS_file_search(dict_name):
    # list.pop() list.append()这两个方法就可以实现栈维护功能
    stack = []
    result_txt = []
    stack.append(dict_name)
    while len(stack) != 0:  # 栈空代表所有目录均已完成访问
        temp_name = stack.pop()
        try:
            temp_name2 = os.listdir(temp_name)  # list ["","",...]
            for eve in temp_name2:
                stack.append(temp_name + "\\" + eve)  # 维持绝对路径的表达
        except NotADirectoryError:
            result_txt.append(temp_name)
    return result_txt

def prue_text(text):
    copytext = [eve for eve in text]
    # corpus 存储语料库，其中以每一个自然段为一个分割
    # 汉语、标点符号都当作有效的字符
    regex_str = ".*?([^\u4E00-\u9FA5]).*?"
    english = "[a-zA-Z]"
    symbol = []
    for j in range(len(copytext)):
        copytext[j] = re.sub(english, "", copytext[j])
        symbol += re.findall(regex_str, copytext[j])
    count_ = Counter(symbol)
    count_symbol = count_.most_common()
    noise_symbol = []
    for eve_tuple in count_symbol:
        if eve_tuple[1] < 200:
            noise_symbol.append(eve_tuple[0])
    noise_number = 0
    for l in range(len(copytext)):
        for noise in noise_symbol:
            copytext[l].replace(noise, "")
            noise_number += 1
        copytext[l] = " ".join(jieba.lcut(copytext[l]))
    print("完成的噪声数据替换点：", noise_number)
    return copytext


path_list = DFS_file_search(r".\小说全集")

# path_list 为包含所有小说文件的路径列表
corpus = dict()
for path in path_list:
    with open(path, "r", encoding="ANSI") as file:
        text = [line.strip("\n").replace("\u3000", "").replace("\t", "") for line in file][3:]
        text = prue_text(text)
        corpus[path.split("\\")[-1].split(".")[0]] = text

# 把16部小说保存成训练集
train_path = r"训练集/train.json"
with open(train_path, "w") as f:
    json.dump(corpus, f)






