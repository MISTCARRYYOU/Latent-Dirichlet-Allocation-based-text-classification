import random
import json

with open("训练集/train.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)
# 随机抽取测试片段，保存到测试集

# 一个小说选2/5个片段
test = dict()
random_len = [5000, 10000, 20000, 40000, 70000, 100000]
large = ["鹿鼎记", "射雕英雄传", "神雕侠侣", "天龙八部", "笑傲江湖", "倚天屠龙记"]
for key in corpus.keys():
    one_string = "".join(corpus[key])
    add = 0
    # 设置长度
    # length1 = random_len[random.randint(0, 5)]
    # length2 = random_len[random.randint(0, 2)]
    length = 20000
    length1 = 20000
    length2 = 20000

    if key in large:  # 10个片段

        for i in range(10):


            assert length1 < len(one_string)
            if add % len(one_string) > len(one_string)-length1:
                add =0
            start = random.randint(add % len(one_string), len(one_string)-length1)
            add += length1
            test[key+"_"+str(length1)+"_"+str(start)] = one_string[start:start+length1]
    else:
        for i in range(4):  # 4个片段

            # print(key, length2, len(one_string))
            assert length2 < len(one_string)
            if add % len(one_string) > len(one_string)-length2:
                add = 0
            start = random.randint(add % len(one_string), len(one_string) - length2)
            add += length2
            test[key + "_" + str(length2) + "_" + str(start)] = one_string[start:start + length2]
for key in test.keys():
    print(key)
test_path = r"测试集/test" + str(length) + ".json"
with open(test_path, "w") as f:
    json.dump(test, f)

with open("测试集\\test.txt", "w", encoding="utf-8") as f:
    for key in test.keys():
        f.write(test[key])
        f.write(r"----")