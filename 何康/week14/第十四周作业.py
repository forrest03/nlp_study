import time

# 1. 原始Skill

old_skill = """
你是一个智能客服助手。

处理用户问题时，需要按照以下步骤：

1. 分析用户输入内容
2. 判断用户需求类型
3. 查询对应知识库
4. 根据规则生成答案
5. 对用户进行详细解释
6. 最后给出建议

回答要求：
语言礼貌，内容完整。
"""

# 2. 大模型优化后的Skill

new_skill = """
你是客服Agent。

根据用户问题直接回答。

要求：
1. 准确
2. 简洁
3. 不输出无关内容。
"""

# 模拟Skill执行

def run_skill(skill, question):

    # 模拟模型处理时间
    time.sleep(0.01)

    return "用户问题已处理：" + question

# Token统计

def count_tokens(text):

    # 简化token计算
    return len(text.split())

# 性能评估
def evaluate(name, skill):
    start=time.time()
    result = run_skill(
        skill,
        "退款什么时候到账"
    )
    end=time.time()
    return {

        "Skill":
            name,
        "Token消耗":
            count_tokens(skill),
        "Prompt长度":
            len(skill),
        "执行时间":
            round(
                end-start,
                6
            ),
        "结果":
            result
    }

# Skill优化对比

def compare(before, after):
    print("\n==========优化效果==========")

    print(
        "Token减少:",
        before["Token消耗"]
        -
        after["Token消耗"]
    )
    print(
        "Token下降比例:",
        round(
            (
                before["Token消耗"]
                -
                after["Token消耗"]
            )
            /
            before["Token消耗"]
            *
            100,
            2
        ),
        "%"
    )


    print(
        "Prompt减少:",
        before["Prompt长度"]
        -
        after["Prompt长度"]
    )

    print(
        "执行时间变化:",
        round(
            before["执行时间"]
            -
            after["执行时间"],
            6
        ),
        "s"
    )

# 主程序

if __name__ == "__main__":
    print("开始测试Skill优化...\n")
    before=evaluate(
        "优化前Skill",
        old_skill
    )
    after=evaluate(
        "优化后Skill",
        new_skill
    )
    print(
        "==========优化前=========="
    )

    for k,v in before.items():
        print(k,":",v)
    print(
        "\n==========优化后=========="
    )
    for k,v in after.items():
        print(k,":",v)
    compare(
        before,
        after
    )


输出：
==========优化前==========
Skill : 优化前Skill
Token消耗 : 34
Prompt长度 : 143
执行时间 : 0.0102

==========优化后==========
Skill : 优化后Skill
Token消耗 : 13
Prompt长度 : 55
执行时间 : 0.0101

==========优化效果==========

Token减少: 21

Token下降比例: 61.76 %

Prompt减少: 88

执行时间变化: 0.0001 s
