from shlex import join


class Vocab:
    
    def __init__(self,texts):
        self.texts = texts

    #构建词表
    def build_vocab(self):
        #构建词表
        c2i = {}
        i2c = {}
        #先把词表构建成char列表，用set不会重复字符
        char_list = set(self.texts)
        #print(char_list)
        #逐个便利字符，构建c2i字标，即key=字 value=index 
        for i,c in enumerate(char_list):
            c2i[c] = i
        #构建i2c字标，即key=index value=字
        for k,v in c2i.items():
            i2c[v] = k
        
        return i2c,c2i
