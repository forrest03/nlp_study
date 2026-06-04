# 第六部分-类与面向对象编程 作业说明
# 
# 本部分作业共4道题，从简单到复杂，帮助巩固以下知识点：
# - 类的定义和创建实例
# - 类属性与实例属性
# - __init__方法和self参数
# - 实例方法的定义和调用
# - 方法中使用属性
#
# 注意：本部分作业可以使用之前学过的所有知识
# - 变量、数据类型、运算符、输入输出
# - 条件语句、循环结构
# - 列表、字典等数据结构
# - 函数
# - 类
#
# ============================================================================
# 作业一：类的基础 - 学生类
# ============================================================================
#
# 任务描述：
# 定义一个Student类，包含以下内容：
# 1. __init__方法：初始化学生的姓名、年龄、学号
# 2. display_info方法：显示学生的基本信息
# 3. set_score方法：设置学生的数学、语文、英语成绩
# 4. calculate_average方法：计算学生的平均分
# 5. get_grade方法：根据平均分返回等级
#
# 在主程序中：
# - 创建至少2个Student实例
# - 调用各个方法测试功能
# - 显示所有学生的信息
#
# 要求：
# - 正确使用self参数
# - 在方法中访问和修改实例属性
# - 添加适当的注释
#
# ============================================================================

class Student:
    def __init__(self, name, age, number):
        self.name = name
        self.age = age
        self.number = number
        self.math = 0
        self.chinese = 0
        self.english = 0
    def display_info(self):
        print("姓名:", self.name, "年龄:", self.age, "学号:", self.number)
        print("数学:", self.math, "语文:", self.chinese, "英语:", self.english)
    def set_score(self, math, chinese, english):
        self.math = math
        self.chinese = chinese
        self.english = english
    def calculate_average(self):
        return (self.math + self.chinese + self.english) / 3
    def get_grade(self):
        avg = self.calculate_average()
        if avg >= 90:
            return "优秀"
        elif avg >= 80:
            return "良好"
        elif avg >= 70:
            return "中等"
        elif avg >= 60:
            return "及格"
        else:
            return "不及格"
s1 = Student("小明", 18, "1001")
s2 = Student("小红", 17, "1002")
s1.set_score(88, 92, 95)
s2.set_score(75, 80, 78)
s1.display_info()
print("平均分:", round(s1.calculate_average(), 1), "等级:", s1.get_grade())
print("-" * 20)
s2.display_info()
print("平均分:", round(s2.calculate_average(), 1), "等级:", s2.get_grade())


# 作业二：类的方法 - 银行账户类
# ============================================================================
#
# 任务描述：
# 定义一个BankAccount类，模拟银行账户，包含以下内容：
# 1. __init__方法：初始化账户号、账户名、初始余额（默认0）
# 2. deposit方法：存款（增加余额）
# 3. withdraw方法：取款（减少余额，需要检查余额是否充足）
# 4. get_balance方法：查询余额
# 5. display_info方法：显示账户信息
#
# 在主程序中：
# - 创建一个账户
# - 进行多次存款和取款操作
# - 显示账户信息
#
# 要求：
# - 使用self访问和修改属性
# - 处理余额不足的情况
# - 添加适当的注释
#
# ============================================================================

class BankAccount:
    def __init__(self, account, name, balance = 0):
        self.account = account
        self.name = name
        self.balance = balance
    def deposit(self, money):
        if money > 0:
            self.balance = self.balance + money
            print(f" 存入：{money}，当前余额：{self.balance}")
        else:
            print("需要存入正数")
    def withdraw(self, money):
        if self.balance < money:
            print("钱不够")
        else:
            self.balance = self.balance - money
            print(f"取出：{money}，当前余额：{self.balance}")
    def get_balance(self):
        print(self.balance)
    def display_info(self):
        print(f"账户号：{self.account}")
        print(f"账户名：{self.name}")
        print(f"账户余额：{self.balance}")
account = BankAccount("123456", "张三", 1000)
account.display_info()
account.deposit(500)
account.deposit(300)
account.withdraw(400)
account.withdraw(2000)
account.display_info()    


# 作业三：类的综合应用 - 购物车类
# ============================================================================
#
# 任务描述：
# 定义一个ShoppingCart类，模拟购物车，包含以下内容：
# 1. __init__方法：初始化购物车（空列表）
# 2. add_item方法：添加商品到购物车（商品信息用字典存储）
# 3. remove_item方法：从购物车移除商品
# 4. calculate_total方法：计算购物车总价
# 5. display_cart方法：显示购物车内容
# 6. clear_cart方法：清空购物车
#
# 在主程序中：
# - 创建一个购物车实例
# - 添加多个商品
# - 显示购物车
# - 计算总价
# - 移除某个商品
# - 再次显示购物车和总价
#
# 要求：
# - 使用列表和字典存储数据
# - 在方法中使用self访问属性
# - 添加适当的注释
#
# ============================================================================

class ShoppingCart:
    def __init__(self):
        self.cart = []
    def add_item(self, item):
        self.cart.append(item)
    def remove_item(self, item_name):
        for i in self.cart:
            if i['name'] == item_name:
                self.cart.remove(i)
                break
    def calculate_total(self):
        total = 0
        for item in self.cart:
            total += item['price']
        return total
    def display_cart(self):
        if not self.cart:
            print("购物车为空")
            return
        print("购物车商品：")
        for item in self.cart:
            print(f"{item['name']} ￥{item['price']}")
    def clear_cart(self):
        self.cart.clear()

my_cart = ShoppingCart()
my_cart.add_item({'name': '牛奶', 'price': 11})
my_cart.add_item({'name': '面包', 'price': 22})
my_cart.add_item({'name': '鸡蛋', 'price': 33})
my_cart.display_cart()
print("总价：", my_cart.calculate_total())
my_cart.remove_item('面包')
print("\n移除面包后：")
my_cart.display_cart()
print("总价：", my_cart.calculate_total())



# 作业四：综合应用 - 学生管理系统（类版）
# ============================================================================
#
# 任务描述：
# 定义一个Student类和一个StudentManager类：
#
# Student类：
# 1. __init__方法：初始化学生信息
# 2. calculate_average方法：计算平均分
# 3. get_grade方法：获取等级
# 4. display_info方法：显示学生信息
#
# StudentManager类：
# 1. __init__方法：初始化学生列表（空列表）
# 2. add_student方法：添加学生
# 3. find_student方法：查找学生
# 4. display_all方法：显示所有学生
# 5. get_class_average方法：计算班级平均分
#
# 在主程序中：
# - 创建StudentManager实例
# - 添加多个学生
# - 显示所有学生信息
# - 查找并显示指定学生
# - 显示班级平均分
#
# 要求：
# - 使用类组织代码
# - 一个类管理多个对象
# - 添加详细的注释
#
# ============================================================================

class Student:
    def __init__(self, name, scores):
        self.name = name
        self.scores = scores
    def calculate_average(self):
        return sum(self.scores) / len(self.scores)
    def get_grade(self):
        avg = self.calculate_average()
        if avg >= 90:
            return 'A'
        elif avg >= 80:
            return 'B'
        elif avg >= 70:
            return 'C'
        else:
            return 'D'
    def display_info(self):
        avg = self.calculate_average()
        grade = self.get_grade()
        print(f"姓名：{self.name}，分数：{self.scores}，平均分：{avg:.1f}，等级：{grade}")
class StudentManager:
    def __init__(self):
        self.students = []
    def add_student(self, student):
        self.students.append(student)
    def find_student(self, name):
        for s in self.students:
            if s.name == name:
                return s
        return None
    def display_all(self):
        print("\n             所有学生信息")
        for s in self.students:
            s.display_info()
    def get_class_average(self):
        total = 0
        for s in self.students:
            total += s.calculate_average()
        return total / len(self.students)
manager = StudentManager()
manager.add_student(Student("张三", [85, 92, 88]))
manager.add_student(Student("李四", [78, 82, 80]))
manager.add_student(Student("王五", [95, 98, 96]))
manager.display_all()
name = "李四"
s = manager.find_student(name)
if s:
    print(f"\n查找结果：{name}")
    s.display_info()
class_avg = manager.get_class_average()
print(f"\n班级平均分：{class_avg:.1f}")


# 提交要求
# ============================================================================
#
# 1. 每个作业创建一个独立的.py文件，命名为：作业一.py、作业二.py 等
# 2. 代码要符合Python代码规范（命名、缩进、注释等）
# 3. 确保代码可以正常运行
# 4. 完成后可以对照参考答案检查自己的代码
#
# 祝学习愉快！








