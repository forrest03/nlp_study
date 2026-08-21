整体流程：
已有文件(txt/pdf/doc)
        |
        ↓
文本读取
        |
        ↓
文本切分
        |
        ↓
TF-IDF文本向量化
        |
        ↓
建立知识库
        |
        ↓
用户输入问题
        |
        ↓
问题向量化
        |
        ↓
计算余弦相似度
        |
        ↓
返回最相似文本作为答案


代码：
# ==========================
# 1.读取知识库文件
# ==========================
def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

# ==========================
# 2.文本切分
# ==========================
def split_text(text):
    # 根据句号切分
    sentences = text.replace("\n", "").split("。")
    result = []
    for s in sentences:
        if len(s.strip()) > 0:
            result.append(s)
    return result

# ==========================
# 3.中文分词
# ==========================
def chinese_tokenize(text):
    words = jieba.cut(text)
    return " ".join(words)
  
# ==========================
# 4.建立知识库
# ==========================
text = load_file("knowledge.txt")
documents = split_text(text)

# 分词处理
documents_cut = []
for doc in documents:

    documents_cut.append(
        chinese_tokenize(doc)
    )
# TF-IDF向量化
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(
    documents_cut
)

# ==========================
# 5.问答函数
# ==========================
def answer_question(question):
    # 问题分词
    question_cut = chinese_tokenize(question)

    # 转换向量
    q_vector = vectorizer.transform(
        [question_cut]
    )

    # 计算相似度
    similarity = cosine_similarity(
        q_vector,
        doc_vectors
    )

    # 找最大相似度位置
    index = similarity.argmax()
    score = similarity[0][index]

    # 判断是否匹配
    if score < 0.1:
        return "知识库中没有找到相关答案"
    else:
        return documents[index]
# ==========================
# 6.交互问答
# ==========================
print("===================")
print("简单问答系统")
print("输入exit退出")
print("===================")

while True:
    question = input("\n请输入问题:")
    if question=="exit":
        break

    result = answer_question(question)

    print("\n回答:")
    print(result)


运行效果：
请输入问题:
什么是人工智能
回答:
人工智能是研究如何让计算机模拟人类智能行为的一门科学

Python是什么
回答:
Python是一种高级编程语言，具有简单易学、功能强大的特点
