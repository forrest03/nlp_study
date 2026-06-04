# 第五部分-函数 作业说明
# 
# 本部分作业共5道题，从简单到复杂，帮助巩固以下知识点：
# - 函数的定义和调用
# - 函数参数（无参数、单个参数、多个参数）
# - 函数返回值
# - 默认参数和关键字参数
# - 局部变量与全局变量
#
# 注意：本部分作业可以使用之前学过的所有知识
# - 变量、数据类型、运算符、输入输出
# - 条件语句、循环结构
# - 列表、字典等数据结构
# - 函数
#
# ============================================================================
# 作业一：函数基础 - 计算器函数
# ============================================================================
#
# 任务描述：
# 编写一个程序，定义以下函数：
# 1. add(a, b)：计算两个数的和
# 2. subtract(a, b)：计算两个数的差
# 3. multiply(a, b)：计算两个数的积
# 4. divide(a, b)：计算两个数的商（注意处理除数为0的情况）
# 5. calculate_bmi(weight, height)：计算BMI指数（体重除以身高的平方）
#
# 在主程序中：
# - 测试每个函数，调用并输出结果
# - 提示用户输入两个数字，调用add函数计算和
# - 提示用户输入体重和身高，调用calculate_bmi函数计算BMI
#
# 要求：
# - 每个函数都要有清晰的注释说明
# - 使用return语句返回结果
# - 处理除数为0的情况
# - 添加适当的注释
#
# ============================================================================

def add(a, b):
    # 计算两个数的和
    result = a + b
    return result
def subtract(a, b):
    # 计算两个数的差
    result = a - b
    return result
def multiply(a, b):
    # 计算两个数的积
    result = a * b
    return result
def divide(a, b):
    if b == 0:
        print("错误！除数不能为0")
        return None
    else:
        result = a / b
        return result
def calculate_bmi(weight, height):
    # BMI = 体重 / 身高的平方
    bmi = weight / (height ** 2)
    return bmi

print("5 + 3 =", add(5, 3))
print("5 - 3 =", subtract(5, 3))
print("5 * 3 =", multiply(5, 3))
print("5 / 3 =", divide(5, 3))
print("6 / 0 =", divide(6, 0))

print("\n用户输入计算加法")
num1 = float(input("请输入第一个数字："))
num2 = float(input("请输入第二个数字："))
he = add(num1, num2)
print("两个数的和是：", he)

print("\n计算BMI")
weight = float(input("请输入体重(kg)："))
height = float(input("请输入身高(m)："))
bmi_result = calculate_bmi(weight, height)
print("你的BMI指数是：", bmi_result)
print("#" * 40)

# 作业二：函数参数与返回值 - 成绩处理函数
# ============================================================================
#
# 任务描述：
# 编写一个程序，定义以下函数：
# 1. calculate_average(scores)：接收一个成绩列表，计算并返回平均分
# 2. find_max_min(scores)：接收一个成绩列表，返回最高分和最低分（返回两个值）
# 3. count_pass_fail(scores, pass_score=60)：接收成绩列表和及格分数线（默认60），返回及格人数和不及格人数
# 4. get_grade(score)：接收一个分数，返回等级（优秀/良好/中等/及格/不及格）
# 5. process_scores(scores)：接收成绩列表，调用以上函数，返回处理结果（字典形式）
#
# 在主程序中：
# - 创建一个成绩列表
# - 调用process_scores函数处理成绩
# - 输出格式化的成绩报告
#
# 要求：
# - 使用默认参数（pass_score=60）
# - 函数返回多个值（使用元组）
# - 函数返回字典
# - 添加详细的注释
#
# ============================================================================

def calculate_average(scores):
    total = sum(scores)
    average = total / len(scores)
    return average
def find_max_min(scores):
    max_score = max(scores)
    min_score = min(scores)
    return max_score, min_score
def count_pass_fail(scores, pass_score=60):
    pass_count = 0
    fail_count = 0
    for score in scores:
        if score >= pass_score:
            pass_count = pass_count + 1
        else:
            fail_count = fail_count + 1
    return pass_count, fail_count
def get_grade(score):
    if score >= 90:
        return "优秀"
    elif score >= 80:
        return "良好"
    elif score >= 70:
        return "中等"
    elif score >= 60:
        return "及格"
    else:
        return "不及格"
def process_scores(scores):
    avg = calculate_average(scores)
    max_s, min_s = find_max_min(scores)
    pass_num, fail_num = count_pass_fail(scores)
    result = {
        "平均分": avg,
        "最高分": max_s,
        "最低分": min_s,
        "及格人数": pass_num,
        "不及格人数": fail_num
    }
    return result
scores = [85, 92, 73, 66, 55, 78, 90, 82]
report = process_scores(scores)
print("平均分：", report["平均分"])
print("最高分：", report["最高分"])
print("最低分：", report["最低分"])
print("及格人数：", report["及格人数"])
print("不及格人数：", report["不及格人数"])
print("#" * 40)

# 作业三：函数综合应用 - 学生管理系统
# ============================================================================
#
# 任务描述：
# 编写一个学生管理系统，定义以下函数：
# 1. create_student(name, age, scores)：创建并返回一个学生字典
# 2. calculate_student_average(student)：计算学生的平均分并更新字典
# 3. get_student_grade(student)：根据平均分判断等级并更新字典
# 4. display_student_info(student)：格式化显示学生信息
# 5. add_student(students_list, name, age, scores)：向学生列表添加新学生
# 6. find_student(students_list, name)：在学生列表中查找指定姓名的学生
# 7. get_class_average(students_list)：计算班级平均分
#
# 在主程序中：
# - 创建一个空的学生列表
# - 使用循环添加至少3个学生
# - 显示所有学生信息
# - 查找并显示指定学生的信息
# - 显示班级平均分
#
# 要求：
# - 使用函数组织代码
# - 函数之间可以相互调用
# - 使用列表和字典存储数据
# - 添加详细的注释
#
# ============================================================================

def create_student(name, age, scores):
    student = {
        "name": name,
        "age": age,
        "scores": scores
    }
    return student
def calculate_student_average(student):
    total = sum(student["scores"])
    average = total / len(student["scores"])
    student["average"] = average
def get_student_grade(student):
    avg = student["average"]
    if avg >= 90:
        grade = "优秀"
    elif avg >= 80:
        grade = "良好"
    elif avg >= 70:
        grade = "中等"
    elif avg >= 60:
        grade = "及格"
    else:
        grade = "不及格"
    student["grade"] = grade
def display_student_info(student):
    print("姓名：", student["name"])
    print("年龄：", student["age"])
    print("成绩：", student["scores"])
    print("平均分：%.2f" % student["average"])
    print("等级：", student["grade"])
    print("-" * 20)
def add_student(students_list, name, age, scores):
    stu = create_student(name, age, scores)
    calculate_student_average(stu)
    get_student_grade(stu)
    students_list.append(stu)
def find_student(students_list, name):
    for s in students_list:
        if s["name"] == name:
            return s
    return None
def get_class_average(students_list):
    total = 0
    for s in students_list:
        total += s["average"]
    return total / len(students_list)
students = []
for i in range(3):
    print("请输入第%d个学生信息：" % (i + 1))
    name = input("姓名：")
    age = int(input("年龄："))
    s1 = int(input("科目1成绩："))
    s2 = int(input("科目2成绩："))
    s3 = int(input("科目3成绩："))
    scores = [s1, s2, s3]
    add_student(students, name, age, scores)
    print("-" * 20)
print("所有学生信息")
for s in students:
    display_student_info(s)
find_name = input("请输入要查找的学生姓名：")
stu = find_student(students, find_name)
if stu:
    print("查找结果")
    display_student_info(stu)
else:
    print("未找到该学生")
class_avg = get_class_average(students)
print("班级平均分：%.2f" % class_avg)


# 作业四：默认参数与关键字参数
# ============================================================================
#
# 任务描述：
# 编写一个程序，定义以下函数：
# 1. greet(name, greeting="你好", punctuation="！")：
#    使用默认参数，打印问候语
# 2. calculate_discount_price(price, discount=0.1, tax=0.0)：
#    计算折扣后的价格（考虑税费）
# 3. create_student(name, age, grade="未定", city="未知")：
#    创建学生信息字典，使用默认参数
# 4. print_info(title, items, separator=", ", end="\n")：
#    格式化打印信息，使用关键字参数
#
# 在主程序中：
# - 测试每个函数，使用默认参数和提供参数两种情况
# - 演示关键字参数的使用
# - 输出结果
#
# 要求：
# - 理解默认参数的作用
# - 理解关键字参数的使用
# - 添加详细的注释
#
# ============================================================================

#  问候函数
def greet(name, greeting="你好", punctuation="！"):
    print(greeting, name, punctuation, sep="")
# 计算折扣后价格（可算税费）
def calculate_discount_price(price, discount=0.1, tax=0.0):
    discounted = price * (1 - discount)
    final = discounted * (1 + tax)
    return final
# 创建学生字典
def create_student(name, age, grade="未定", city="未知"):
    student = {
        "姓名": name,
        "年龄": age,
        "年级": grade,
        "城市": city
    }
    return student
def print_info(title, items, separator=", ", end="\n"):
    print(title, end=" ")
    print(separator.join(items), end=end)
greet("小明")                     
greet("小红", greeting="早上好")   
greet("小李", punctuation="!!!")  
print(calculate_discount_price(100))                
print(calculate_discount_price(100, discount=0.2))  
print(calculate_discount_price(100, tax=0.1))      
s1 = create_student("小明", 18)
s2 = create_student("小红", 17, grade="高一", city="北京")
print(s1)
print(s2)
print_info("科目：", ["语文", "数学", "英语"])
print_info("爱好：", ["看书", "打球"], separator=" | ")

# 作业五：综合应用 - 计算器程序（函数版）
# ============================================================================
#
# 任务描述：
# 将之前的购物车系统改造成使用函数的版本，定义以下函数：
# 1. display_menu()：显示菜单
# 2. show_products(products)：显示所有商品
# 3. add_to_cart(cart, products, product_name, quantity)：添加商品到购物车
# 4. show_cart(cart)：显示购物车内容
# 5. calculate_total(cart)：计算购物车总价
# 6. clear_cart(cart)：清空购物车
# 7. main()：主函数，包含主循环和菜单逻辑
#
# 在主程序中：
# - 调用main()函数启动程序
# - 使用函数组织代码，使程序更模块化
#
# 要求：
# - 使用函数重构代码
# - 每个功能对应一个函数
# - 使用main()函数作为程序入口
# - 代码结构清晰，易于维护
# - 添加详细的注释
#
# ============================================================================

# 显示菜单
def display_menu():
    print("购物车菜单\n1. 查看所有商品\n2. 添加商品到购物车\n3. 查看购物车\n4. 计算购物车总价\n5. 清空购物车\n6. 退出")
    print("-" * 20)
# 显示所有商品
def show_products(products):
    print("商品列表：")
    for p in products:
        print("名称：", p["名称"], " 价格：", p["价格"], " 库存：", p["库存"])
# 添加商品到购物车
def add_to_cart(cart, products, product_name, quantity):
    find_p = None
    for p in products:
        if p["名称"] == product_name:
            find_p = p
            break
    if find_p == None:
        print("商品不存在！")
        return
    if find_p["库存"] >= quantity:
        item = {"名称": product_name, "价格": find_p["价格"], "数量": quantity}
        cart.append(item)
        print("添加成功！")
    else:
        print("库存不足！")
# 显示购物车
def show_cart(cart):
    if len(cart) == 0:
        print("购物车是空的")
        return
    print("购物车商品：")
    for item in cart:
        print(item["名称"], " 单价：", item["价格"], " 数量：", item["数量"])
# 计算总价
def calculate_total(cart):
    total = 0
    for item in cart:
        total += item["价格"] * item["数量"]
    return total
# 清空购物车
def clear_cart(cart):
    cart.clear()
    print("购物车已清空")
def main():
    products = [
        {"名称": "苹果", "价格": 5, "库存": 100},
        {"名称": "香蕉", "价格": 3, "库存": 80},
        {"名称": "牛奶", "价格": 4, "库存": 50},
        {"名称": "面包", "价格": 6, "库存": 30},
        {"名称": "矿泉水", "价格": 2, "库存": 200}
    ]
    cart = []
    while True:
        display_menu()
        choice = input("请输入选择：")
        if choice == "1":
            show_products(products)
        elif choice == "2":
            name = input("输入商品名称：")
            num = int(input("输入数量："))
            add_to_cart(cart, products, name, num)
        elif choice == "3":
            show_cart(cart)
        elif choice == "4":
            total = calculate_total(cart)
            print("总价：", total)
        elif choice == "5":
            clear_cart(cart)
        elif choice == "6":
            print("谢谢使用，再见！")
            break
        else:
            print("输入错误，请重新选择！")
main()


# 提交要求
# ============================================================================
#
# 1. 每个作业创建一个独立的.py文件，命名为：作业一.py、作业二.py 等
# 2. 代码要符合Python代码规范（命名、缩进、注释等）
# 3. 确保代码可以正常运行
# 4. 完成后可以对照参考答案检查自己的代码
#
# 祝学习愉快！








