# 第九部分-错误处理与异常处理 作业说明
#
# 本部分作业共4道题，从简单到复杂，帮助巩固以下知识点：
# - 使用try-except捕获和处理异常
# - 处理不同类型的异常（ValueError、TypeError、FileNotFoundError等）
# - 使用try-except-else-finally
# - 异常处理在实际应用中的使用
#
# 注意：本部分作业可以使用之前学过的所有知识
# - 变量、数据类型、运算符、输入输出
# - 条件语句、循环结构
# - 列表、字典等数据结构
# - 函数
# - 类
# - 文件操作
# - 异常处理
#
# ============================================================================
# 作业一：基础异常处理 - 安全的数值计算
# ============================================================================
#
# 任务描述：
# 编写一个程序，实现安全的数值计算功能：
# 1. 编写一个safe_divide函数，实现安全的除法运算
#    - 接收两个参数：被除数和除数
#    - 使用try-except捕获ZeroDivisionError（除数为0）
#    - 使用try-except捕获TypeError（参数不是数字）
#    - 如果正常计算，返回结果
#    - 如果出错，打印错误信息并返回None
# 2. 编写一个safe_power函数，计算一个数的幂
#    - 接收两个参数：底数和指数
#    - 使用异常处理确保参数是数字
#    - 返回计算结果
# 3. 在主程序中测试这些函数，包括正常情况和异常情况
#
# 要求：
# - 使用try-except捕获具体的异常类型
# - 提供有意义的错误信息
# - 测试多种异常情况
# - 添加适当的注释
#
# ============================================================================


def safe_divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("除数不能为0！")
        return None
    except TypeError:
        print("输入必须是数字！")
        return None

def safe_power(base, exponent):
    try:
        result = base**exponent
        return result
    except TypeError:
        print("底数和指数必须都是数字")
        return None

print("safe_divide 函数")
print(safe_divide(10, 2))
print(safe_divide(10, 0))
print(safe_divide("10", 2))
print("测试 safe_power 函数")
print(safe_power(2, 3))
print(safe_power(5, "2"))
print(safe_power("3", 2))


# 作业二：用户输入验证 - 安全的用户输入处理
# ============================================================================
#
# 任务描述：
# 编写一个程序，实现安全的用户输入验证功能：
# 1. 编写一个get_positive_int函数
#    - 提示用户输入一个正整数
#    - 使用try-except处理输入错误（ValueError）
#    - 验证输入是否为正整数
#    - 如果输入无效，给出提示并让用户重新输入
#    - 使用循环直到输入有效
# 2. 编写一个get_age函数
#    - 获取用户年龄（0-150之间）
#    - 使用异常处理验证输入
#    - 处理非数字输入和超出范围的情况
# 3. 编写一个get_score函数
#    - 获取用户分数（0-100之间）
#    - 支持小数输入
#    - 使用异常处理验证输入
# 4. 在主程序中测试这些函数
#
# 要求：
# - 使用while循环实现重试机制
# - 捕获ValueError处理输入错误
# - 提供清晰的错误提示
# - 使用KeyboardInterrupt处理用户中断（Ctrl+C）
# - 添加适当的注释
#
# ============================================================================


def get_positive_int(prompt="输入一个正整数："):
    while True:
        try:
            num = int(input(prompt))
            if num > 0:
                return num
            else:
                print("必须是正整数")
        except ValueError:
            print("非有效整数")
        except KeyboardInterrupt:
            print("\n用户中断")
            return None

def get_age():
    while True:
        try:
            age = int(input("输入年龄（0-150）："))
            if 0 <= age <= 150:
                return age
            else:
                print("年龄必须在0到150之间")
        except ValueError:
            print("不是有效数字，请重试！")
        except KeyboardInterrupt:
            print("\n用户中断")
            return None

def get_score():
    while True:
        try:
            score = float(input("输入分数0-100："))
            if 0 <= score <= 100:
                return score
            else:
                print("分数必须在 0 到 100 之间")
        except ValueError:
            print("输入不是有效数字")
        except KeyboardInterrupt:
            print("\n用户中断")
            return None


print("测试 get_positive_int")
num = get_positive_int()
print("输入的正整数是：", num)

print("\n测试 get_age")
age = get_age()
print("你的年龄是：", age)

print("\n测试 get_score")
score = get_score()
print("你的分数是：", score)


# 作业三：文件操作异常处理 - 安全读写文件
# ============================================================================
#
# 任务描述：
# 编写一个程序，实现安全的文件操作功能：
# 1. 编写一个read_file_safe函数
#    - 接收文件名作为参数
#    - 尝试读取文件内容
#    - 使用try-except处理FileNotFoundError（文件不存在）
#    - 使用try-except处理PermissionError（权限不足）
#    - 使用try-except处理UnicodeDecodeError（编码错误）
#    - 使用finally确保资源清理
#    - 返回文件内容，如果出错返回None
# 2. 编写一个write_file_safe函数
#    - 接收文件名和内容作为参数
#    - 尝试写入文件
#    - 处理可能的异常
#    - 返回是否成功
# 3. 编写一个copy_file_safe函数
#    - 接收源文件名和目标文件名
#    - 尝试复制文件内容
#    - 使用异常处理确保操作安全
# 4. 在主程序中测试这些函数
#
# 要求：
# - 使用with语句打开文件
# - 捕获具体的异常类型
# - 提供有意义的错误信息
# - 使用finally进行资源清理
# - 添加适当的注释
#
# ============================================================================


def read_file_safe(filename):
    file = None
    try:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"文件 {filename} 不存在！")
    except PermissionError:
        print(f"没有权限读取 {filename}！")
    except UnicodeDecodeError:
        print(f"文件编码错误，无法读取 {filename}！")
    except Exception as e:
        print(f"错误：{e}")
    finally:
        print(f"资源已清理")
    return None

def write_file_safe(filename, content):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"成功写入：{filename}")
        return True
    except PermissionError:
        print(f"没有权限 {filename}！")
    except Exception as e:
        print(f"写入失败：{e}")
    return False

def copy_file_safe(source, target):
    content = read_file_safe(source)
    if content is None:
        print("复制失败")
        return False
    success = write_file_safe(target, content)
    if success:
        print("文件复制成功！")
    return success

read_file_safe("test.txt")
write_file_safe("test.txt", "11111111111")
copy_file_safe("test.txt", "test1.txt")
read_file_safe("test.txt")


# 作业四：综合应用 - 学生成绩管理系统（异常处理版）
# ============================================================================
#
# 任务描述：
# 编写一个学生成绩管理系统，使用异常处理确保程序的健壮性：
# 1. 定义一个StudentScoreManager类
#    - __init__方法：初始化，加载已有成绩数据（如果存在）
#    - load_scores方法：从JSON文件加载成绩，处理文件不存在等异常
#    - save_scores方法：保存成绩到JSON文件，处理写入异常
#    - add_score方法：添加成绩，验证分数范围（0-100），处理异常
#    - get_average方法：计算平均分，处理学生不存在等异常
#    - display_all方法：显示所有学生成绩
# 2. 在主程序中：
#    - 创建StudentScoreManager实例
#    - 添加多个学生的成绩（包括正常和异常情况）
#    - 查询和显示成绩
#    - 处理各种可能的异常
#
# 要求：
# - 使用类组织代码
# - 合理使用异常处理
# - 处理文件操作异常、数据验证异常等
# - 提供友好的错误提示
# - 使用JSON格式存储数据
# - 添加详细的注释
#
# ============================================================================


import json

class StudentScoreManager:
    def __init__(self):
        self.filename = "scores.json"
        self.scores = self.load_scores()

    def load_scores(self):
        try:
            with open(self.filename, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print("文件未找到")
            return {}
        except json.JSONDecodeError:
            print("json格式错误")
            return {}
        except Exception as e:
            print("异常", e)
            return {}

    def save_scores(self):
        try:
            with open(self.filename, "w", encoding="utf-8") as f:
                json.dump(self.scores, f, ensure_ascii=False, indent=2)
            return True
        except PermissionError:
            print("无写入权限")
            return False
        except Exception as e:
            print("保存失败：", e)
            return False

    def add_score(self, name, score):
        try:
            score = float(score)
            if score < 0 or score > 100:
                print("分数必须在0-100之间")
                return False
            self.scores[name] = score
            self.save_scores()
            print(f"成功添加：{name} -> {score}")
            return True
        except ValueError:
            print("分数必须是数字")
            return False

    def get_average(self):
        try:
            if len(self.scores) == 0:
                print("无学生成绩")
                return 0
            total = sum(self.scores.values())
            avg = total / len(self.scores)
            return round(avg, 2)
        except Exception as e:
            print("计算失败：", e)
            return 0

    def display_all(self):
        print("\n所有学生成绩")
        if not self.scores:
            print("无数据")
            return
        for name, score in self.scores.items():
            print(f"{name}：{score}")

manager = StudentScoreManager()
manager.add_score("小明", 90)
manager.add_score("小红", 85.5)
manager.add_score("小刚", "abc")
manager.add_score("小李", 150)
manager.add_score("小王", -5)
manager.display_all()
print("班级平均分：", manager.get_average())



# 提交要求
# ============================================================================
#
# 1. 每个作业创建一个独立的.py文件，命名为：作业一.py、作业二.py 等
# 2. 代码要符合Python代码规范（命名、缩进、注释等）
# 3. 确保代码可以正常运行
# 4. 完成后可以对照参考答案检查自己的代码
#
# 祝学习愉快！
