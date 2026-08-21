'''
训练数据处理
'''

class Corpus:

    def __init__(self,path):
        self.path = path

    #加载训练数据
    def load_train_data(self):
        path = self.path
        texts = []
        #打开预料文件，把文件整个装进列表中
        with open(path,"r",encoding="utf-8",errors="ignore") as f:
           texts.append(f.read())
        #首尾相连，形成列表中的一大段文字
        return "".join(texts)

