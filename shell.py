# 由于家里电脑是windows的，使用这个python脚本作为shell文件的替代，一件运行所有模型测试结果
# 王霄汉
import os

# 用于直接遍历文件夹并返回文件夹的文件相对路径列表，类似于os.walk
def DFS_file_search(dict_name):
    # list.pop() list.append()这两个方法就可以实现栈维护功能
    stack = []
    result_txt = []
    stack.append(dict_name)
    while len(stack) != 0:  # 栈空代表所有目录均已完成访问
        temp_name = stack.pop()
        try:
            temp_name2 = os.listdir(temp_name) # list ["","",...]
            for eve in temp_name2:
                stack.append(temp_name + "\\" + eve)  # 维持绝对路径的表达
        except NotADirectoryError:
            result_txt.append(temp_name)
    return result_txt

# paths = DFS_file_search("测试集")

# 探索不同数据集
# for path in paths:
#     os.system('python LDAscript.py --k={} --test={}'.format(16, path))

# 探索不同主题k
# ks = list(range(6,20,2)) + list(range(20,100,10))
# for k in ks:
#     os.system('python LDAscript.py --k={} --test={}'.format(k, "测试集/test7500.json"))

# 探索关键词
for i in range(10):
    os.system('python LDAscript.py --k={} --test={} --ischar={}'.format(50, "测试集/test20000.json", True))