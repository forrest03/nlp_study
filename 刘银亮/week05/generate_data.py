import json
import random

# 预定义的一些数据模板
TOPICS = [
    "春天", "夏天", "秋天", "冬天", "爱情", "友情", "亲情", "人生",
    "梦想", "努力", "坚持", "成功", "失败", "学习", "工作", "生活",
    "旅行", "阅读", "音乐", "电影", "美食", "健康", "科技", "自然"
]

RESPONSES = [
    "{}是一个美好的人生主题。",
    "关于{}，我认为它代表着人们对生活的热爱。",
    "{}是人生中不可或缺的一部分。",
    "让我们谈谈{}这个话题。",
    "{}在我们的生活中扮演着重要的角色。",
    "对于{}，每个人都有自己独特的理解。",
    "{}能够给我们带来快乐和满足感。",
    "追求{}是人类的本能之一。",
    "在当今社会，{}变得越来越重要。",
    "{}需要我们用心去体会和珍惜。",
]

POEMS = [
    "春风拂面花自开，\n秋月照人影徘徊。\n夏雨清凉消暑气，\n冬雪皑皑覆尘埃。",
    "山高水长路漫漫，\n云卷云舒任自然。\n人间烟火皆过客，\n心若无尘便是仙。",
    "日出东方照四方，\n月升西阁映寒窗。\n世间万物皆有时，\n且把深情付流光。",
    "一壶浊酒喜相逢，\n古今多少事，尽付笑谈中。\n青山依旧在，几度夕阳红。",
    "桃花流水窅然去，\n别有天地非人间。\n此心安处是吾乡，\n何必归隐山水间。",
]

def generate_sample(index):
    """生成单条样本"""
    topic = random.choice(TOPICS)
    response_template = random.choice(RESPONSES)
    response = response_template.format(topic)

    # 随机决定输出类型
    if random.random() < 0.3:
        output = random.choice(POEMS)
    else:
        output = response

    instructions = [
        f"请谈谈关于{topic}的看法",
        f"什么是{topic}？",
        f"你能帮我解释一下{topic}吗？",
        f"关于{topic}，你有什么想法？",
        f"我想了解一下{topic}",
        f"说说你对{topic}的理解",
        f"{topic}为什么重要？",
        f"如何正确看待{topic}？",
    ]

    return {
        "instruction": random.choice(instructions),
        "output": output
    }

def generate_data(count, output_file):
    """生成数据并写入文件"""
    random.seed(42)  # 保证可复现
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(count):
            sample = generate_sample(i)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"已生成 {count} 条数据到 {output_file}")

if __name__ == "__main__":
    generate_data(1000, "train.jsonl")
    generate_data(500, "eval.jsonl")
    print("数据生成完成!")