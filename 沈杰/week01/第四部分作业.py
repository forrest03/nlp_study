# 第四部分-数据结构 作业说明
#
# 本部分作业共5道题，从简单到复杂，帮助巩固以下知识点：
# - 列表：创建、访问、修改、遍历
# - 元组：创建、访问、与列表的区别
# - 字典：创建、访问、修改、遍历
# - 集合：创建、基本操作
#
# 注意：本部分作业可以使用之前学过的所有知识
# - 变量、数据类型、运算符、输入输出
# - 条件语句、循环结构
# - 列表、元组、字典、集合
#
# ============================================================================
# 作业一：列表操作基础
# ============================================================================
#
# 任务描述：
# 编写一个程序，实现以下功能：
# 1. 创建一个空列表，用于存储学生姓名
# 2. 使用循环提示用户输入5个学生的姓名，并添加到列表中
# 3. 使用for循环遍历列表，打印所有学生姓名
# 4. 计算并打印列表的长度
# 5. 提示用户输入一个要查找的姓名，判断该姓名是否在列表中
# 6. 如果存在，打印该姓名在列表中的位置（索引）
#
# 要求：
# - 使用列表的append()方法添加元素
# - 使用len()函数获取列表长度
# - 使用in关键字判断元素是否存在
# - 使用index()方法查找索引（如果存在）
# - 添加适当的注释
#
# ============================================================================

studentNames = []
# 输入5次学生姓名
for i in range (5):
    name = input("请输入学生姓名:")
    studentNames.append(name)
for name in studentNames:
    print(name)
print(f"列表长度{len(studentNames)}")
search = input("请输入想要查找的学生姓名:")
if search in studentNames:
    print(f"该学生在列表中,位置为{studentNames.index(search)}")
else:
    print("该学生不在列表")
print("#" * 40)

# 作业二：列表综合应用 - 成绩管理系统
# ============================================================================
#
# 任务描述：
# 编写一个成绩管理系统，实现以下功能：
# 1. 创建一个列表存储5个学生的成绩（可以预设或用户输入）
# 2. 计算并输出：总分、平均分、最高分、最低分
# 3. 统计及格人数（成绩>=60）和不及格人数
# 4. 找出所有大于等于90分的成绩，并输出
# 5. 使用循环遍历列表，为每个成绩评定等级：
#    - 90分及以上：优秀
#    - 80-89分：良好
#    - 70-79分：中等
#    - 60-69分：及格
#    - 60分以下：不及格
# 6. 输出格式化的成绩单
#
# 要求：
# - 使用列表的索引访问元素
# - 使用for循环遍历列表
# - 使用条件语句进行判断
# - 使用列表的append()方法（如果需要）
# - 输出格式要清晰美观
#
# ============================================================================

print("预设成绩为[60,70,80,90,100]")
studentsScore = [60,70,80,90,100]
sum = sum(studentsScore)
print(f"总分:{sum}\n平均分:{sum/5}\n最高分:{max(studentsScore)}\n最低分:{min(studentsScore)}\n")
countPass = 0
for i in range(len(studentsScore)):
    if studentsScore[i] >= 60:
        countPass = countPass + 1
print(f"及格人数为{countPass}人,不及格人数为{5 - countPass}人")
print("分数大于90的成绩为:")
for i in studentsScore:
    if i >= 90:
        print(i)
print("----------------\n对每个成绩进行评定,评定结果:")
for i in studentsScore:
    if i >= 90:
        print(f"{i}分,优秀")
    elif i >=80:
        print(f"{i}分,良好")
    elif i >=70:
        print(f"{i}分,中等")
    elif i >=60:
        print(f"{i}分,及格")
    else:
        print(f"{i}分,不及格")
print("#" * 40)

# 作业三：字典操作 - 学生信息管理
# ============================================================================
#
# 任务描述：
# 编写一个学生信息管理程序，实现以下功能：
# 1. 创建一个字典，存储一个学生的信息：
#    - 姓名、年龄、学号、数学成绩、语文成绩、英语成绩
# 2. 计算该学生的总分和平均分，并添加到字典中
# 3. 根据平均分判断等级，并添加到字典中（等级判断规则同作业二）
# 4. 使用for循环遍历字典，打印所有信息
# 5. 提示用户输入要修改的科目和新的成绩，更新字典
# 6. 重新计算总分、平均分和等级
# 7. 输出更新后的学生信息
#
# 要求：
# - 使用字典存储学生信息
# - 使用字典的键访问和修改值
# - 使用字典的items()方法遍历
# - 使用条件语句进行等级判断
# - 添加适当的注释
#
# ============================================================================

student = {
    "name": "张三",
    "age": 18,
    "number": "001",
    "math": 80,
    "chinese": 90,
    "english": 100,
}
student["totalScore"] = student["chinese"] + student["english"] + student["math"]
averageScore = student["totalScore"] / 3
student["averageScore"] = averageScore
if averageScore >= 90:
    student["level"] = "优秀"
elif averageScore >= 80:
        student["level"] = "良好"
elif averageScore >= 70:
    student["level"] = "中等"
elif averageScore >= 60:
    student["level"] = "及格"
else:
    student["level"] = "不及格"
print("学生信息:")
for i in student:
    print(f"         {i}:{student[i]}")
subject = input("可选的科目为:\nenglish\nmath\nchinese\n请输入想要修改的科目:")
score = int(input("请输入新的成绩:"))
if subject in student:
    student[subject] = score
else:
    print("输入的科目不存在")
student["totalScore"] = student["chinese"] + student["english"] + student["math"]
averageScore = student["totalScore"] / 3
student["averageScore"] = averageScore
print("更新后的学生信息", student)
print("#" * 40)

# 作业四：列表与字典结合 - 多学生管理系统
# ============================================================================
#
# 任务描述：
# 编写一个多学生管理系统，实现以下功能：
# 1. 创建一个列表，列表中每个元素是一个字典，存储一个学生的信息
#    （至少包含3个学生的信息）
# 2. 使用for循环遍历列表，打印所有学生的基本信息（姓名、年龄、成绩等）
# 3. 计算所有学生的平均分，并找出平均分最高的学生
# 4. 统计每个等级的人数（优秀、良好、中等、及格、不及格）
# 5. 提示用户输入一个学生姓名，查找并显示该学生的详细信息
# 6. 如果找到，允许用户修改该学生的某科成绩
#
# 要求：
# - 使用列表存储多个字典
# - 使用嵌套循环（外层遍历列表，内层遍历字典）
# - 使用条件语句进行判断和查找
# - 输出格式要清晰美观
# - 添加详细的注释
#
# ============================================================================

students = [
    {"name": "张三", "age": 18, "python": 95, "math": 88},
    {"name": "李四", "age": 19, "python": 72, "math": 65},
    {"name": "王五", "age": 18, "python": 85, "math": 92}
]

print("       所有学生信息")
# 遍历学生列表
for student in students:
    for key, value in student.items():
        print(f"{key}: {value}", end="\t")
    print()  
total_score = 0
student_count = len(students)
max_avg = 0
top_student = ""
print("\n" + "=" * 40)
print("【学生平均分统计】")
print("=" * 50)
for stu in students:
    avg = (stu["python"] + stu["math"]) / 2
    stu["avg"] = avg 
    print(f"{stu['name']} 的平均分：{avg:.2f}")
    total_score += avg
    if avg > max_avg:
        max_avg = avg
        top_student = stu["name"]
# 班级总平均分
class_avg = total_score / student_count
print(f"\n班级总平均分：{class_avg:.2f}")
print(f"平均分最高的学生是：{top_student} ({max_avg:.2f}分)")
excellent = 0   
good = 0        
medium = 0     
pass_ = 0      
fail = 0       
for stu in students:
    if stu["avg"] >= 90:
        excellent += 1
    elif stu["avg"] >= 80:
        good += 1
    elif stu["avg"] >= 70:
        medium += 1
    elif stu["avg"] >= 60:
        pass_ += 1
    else:
        fail += 1
print("\n" + "-" * 20)
print("【成绩等级统计】")
print("\n" + "-" * 20)
print(f"优秀：{excellent} 人")
print(f"良好：{good} 人")
print(f"中等：{medium} 人")
print(f"及格）：{pass_} 人")
print(f"不及格：{fail} 人")
print("根据姓名查找学生 修改成绩")
search_name = input("请输入要查找的学生姓名：")
found_student = None
for stu in students:
    if stu["name"] == search_name:
        found_student = stu
        break
if found_student:
    print("\n 找到学生信息：")
    print(f"姓名：{found_student['name']}")
    print(f"年龄：{found_student['age']}")
    print(f"Python成绩：{found_student['python']}")
    print(f"数学成绩：{found_student['math']}")
    print("\n--- 修改成绩 ---")
    course = input("请输入要修改的科目（python/math）：")
    new_score = int(input(f"请输入{course}新成绩："))
    if course == "python":
        found_student["python"] = new_score
    elif course == "math":
        found_student["math"] = new_score
        
    print("\n 修改成功！最新信息：")
    print(f"Python：{found_student['python']}")
    print(f"数学：{found_student['math']}")
else:
    print("\n 未找到该学生！")
print("\n" + "#" * 40)



# 作业五：综合应用 - 购物车系统
# ============================================================================
#
# 任务描述：
# 编写一个购物车系统，实现以下功能：
# 1. 创建一个商品列表，每个商品是一个字典，包含：
#    - 商品名称、价格、库存数量
#    - 至少包含5个商品
# 2. 创建一个购物车（空列表），用于存储用户选择的商品
# 3. 显示菜单：
#    - 1. 查看所有商品
#    - 2. 添加商品到购物车
#    - 3. 查看购物车
#    - 4. 计算购物车总价
#    - 5. 清空购物车
#    - 6. 退出
# 4. 使用while循环让程序可以重复执行，直到用户选择退出
# 5. 实现各个菜单功能：
#    - 查看商品：遍历商品列表，显示所有商品信息
#    - 添加商品：提示用户输入商品名称和数量，检查库存，添加到购物车
#    - 查看购物车：显示购物车中的所有商品和数量
#    - 计算总价：遍历购物车，计算所有商品的总价
#    - 清空购物车：清空购物车列表
#
# 要求：
# - 使用列表和字典存储数据
# - 使用while循环实现菜单系统
# - 使用for循环遍历列表和字典
# - 使用条件语句进行判断
# - 处理各种边界情况（如商品不存在、库存不足等）
# - 输出格式要清晰美观
# - 添加详细的注释
#
# ============================================================================

goods = [
    {"名称": "苹果", "价格": 5, "库存": 100},
    {"名称": "香蕉", "价格": 3, "库存": 80},
    {"名称": "牛奶", "价格": 4, "库存": 50},
    {"名称": "面包", "价格": 6, "库存": 30},
    {"名称": "矿泉水", "价格": 2, "库存": 200}
]
cart = []
while True:
    print("""====== 购物车菜单 ======
1. 查看所有商品
2. 添加商品到购物车
3. 查看购物车
4. 计算购物车总价
5. 清空购物车
6. 退出
========================""")
    choice = input("请输入你的选择：")
    if choice == "1":
        print("商品列表：")
        for g in goods:
            print("名称：", g["名称"], " 价格：", g["价格"], " 库存：", g["库存"])
    elif choice == "2":
        name = input("请输入要添加的商品名称：")
        num = int(input("请输入要购买的数量："))
        find_good = None
        for g in goods:
            if g["名称"] == name:
                find_good = g
                break
        if find_good == None:
            print("商品不存在！")
        else:
            if find_good["库存"] >= num:
                cart_item = {"名称": name, "价格": find_good["价格"], "数量": num}
                cart.append(cart_item)
                print("添加成功！")
            else:
                print("库存不足！")
    elif choice == "3":
        if len(cart) == 0:
            print("购物车是空的")
        else:
            print("购物车商品：")
            for item in cart:
                print(item["名称"], " 单价：", item["价格"], " 数量：", item["数量"])
    elif choice == "4":
        total = 0
        for item in cart:
            total = total + item["价格"] * item["数量"]
        print("购物车总价是：", total)
    elif choice == "5":
        cart.clear()
        print("购物车已清空")
    elif choice == "6":
        print("谢谢使用，再见！")
        break
    else:
        print("输入错误，请重新选择")



# 提交要求
# ============================================================================
#
# 1. 每个作业创建一个独立的.py文件，命名为：作业一.py、作业二.py 等
# 2. 代码要符合Python代码规范（命名、缩进、注释等）
# 3. 确保代码可以正常运行
# 4. 完成后可以对照参考答案检查自己的代码
#
# 祝学习愉快！
