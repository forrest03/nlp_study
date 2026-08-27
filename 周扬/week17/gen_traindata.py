# -*- coding: utf-8 -*-
import json
import random

random.seed(2026)

# 位数与取值区间（严格等于该位数）
def range_of(L):
    return (10 ** (L - 1), 10 ** L - 1) if L > 1 else (1, 9)

def rand_num(L):
    lo, hi = range_of(L)
    return random.randint(lo, hi)

def gen(op, L, n=100):
    res = []
    if op == "加法":
        for _ in range(n):
            a, b = rand_num(L), rand_num(L)
            res.append(ans(op, L, a, b, a + b))
    elif op == "减法":
        for _ in range(n):
            a, b = rand_num(L), rand_num(L)
            if a < b:
                a, b = b, a
            res.append(ans(op, L, a, b, a - b))
    elif op == "乘法":
        for _ in range(n):
            a, b = rand_num(L), rand_num(L)
            res.append(ans(op, L, a, b, a * b))
    elif op == "除法":  # 除数L位，被除数=除数*商，保证整除且被除数不超过4位
        cnt = 0
        while cnt < n:
            divisor = rand_num(L)
            q_hi = 9999 // divisor
            if q_hi < 1:
                continue
            quotient = random.randint(1, q_hi)
            dividend = divisor * quotient
            res.append(ans(op, L, dividend, divisor, quotient))
            cnt += 1
    return res

SYMBOL = {"加法": "+", "减法": "-", "乘法": "×", "除法": "÷"}

def ans(op, L, x, y, r):
    expr = f"{x} {SYMBOL[op]} {y}"
    return {
        "位数": f"{L}位",
        "type": op,
        "prompt": f"计算：{expr} = ?",
        "expression": expr,
        "answer": str(r)
    }

ops = ["加法", "减法", "乘法", "除法"]
samples = []
for op in ops:
    for L in range(1, 5):
        samples.extend(gen(op, L))

# 统计校验
counts = {}
for s in samples:
    key = (s["位数"], s["type"])
    counts[key] = counts.get(key, 0) + 1
# 校验答案是否正确（自检）
ok = True
for s in samples:
    ex = s["expression"]
    x, o, y = ex.split(" ")
    x, y = int(x), int(y)
    r = {"+": x + y, "-": x - y, "×": x * y, "÷": x // y}[o]
    if r != int(s["answer"]):
        ok = False
        break
print("所有答案校验通过:", ok)
print("总条数:", len(samples))
print("组合数量(位数×运算):")
for k in sorted(counts):
    print("  ", k, counts[k])

with open("traindata.json", "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=2)
print("已写入 traindata.json")