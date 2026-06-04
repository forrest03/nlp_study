# 第八部分-HTTP调用与API基础 作业说明
# 
# 本部分作业共4道题，从简单到复杂，帮助巩固以下知识点：
# - HTTP的基本概念
# - 使用requests库发送HTTP请求
# - GET请求和POST请求
# - 处理API响应（JSON格式）
# - 错误处理
#
# 注意：
# - 需要先安装requests库：pip install requests
# - 可以使用免费的测试API（如JSONPlaceholder）
# - 本部分作业可以使用之前学过的所有知识
#
# ============================================================================
# 作业一：使用requests库 - GET请求
# ============================================================================
#
# 任务描述：
# 编写一个程序，使用requests库完成以下任务：
# 1. 发送GET请求到JSONPlaceholder API获取用户列表
#    URL: https://jsonplaceholder.typicode.com/users
# 2. 打印响应状态码
# 3. 解析JSON响应数据
# 4. 显示前3个用户的姓名和邮箱
# 5. 发送GET请求获取单个用户信息
#    URL: https://jsonplaceholder.typicode.com/users/1
# 6. 显示该用户的详细信息
#
# 要求：
# - 使用requests.get()发送GET请求
# - 使用response.json()解析JSON数据
# - 处理可能的错误（如网络错误）
# - 添加适当的注释
#
# ============================================================================

import requests

response =requests.get('https://jsonplaceholder.typicode.com/users', timeout=5)
print(f"状态码:{response.status_code}")
if response.status_code == 200:
    js = response.json()
    print(f"相应数据{js}")
    print(f"第一个:姓名:{js[0]['username']},email:{js[0]['email']}")
    print(f"第二个:姓名:{js[1]['username']},email:{js[1]['email']}")
    print(f"第三个:姓名:{js[2]['username']},email:{js[2]['email']}")
else:
    print(f"获取失败，状态码：{response.status_code}")

resp = requests.get("https://jsonplaceholder.typicode.com/users/1",timeout=3)
if response.status_code == 200:
    print(f'单个用户请求结果:{resp.json()}')
else:
    print(f"获取失败，状态码：{response.status_code}")


# 作业二：处理API响应 - 数据提取
# ============================================================================
#
# 任务描述：
# 编写一个程序，从JSONPlaceholder API获取数据并进行处理：
# 1. 获取所有帖子（posts）
#    URL: https://jsonplaceholder.typicode.com/posts
# 2. 统计帖子总数
# 3. 找出用户ID为1的所有帖子
# 4. 显示前5个帖子的标题
# 5. 找出标题最长的帖子
#
# 要求：
# - 解析JSON响应
# - 使用循环和条件语句处理数据
# - 输出格式清晰
#
# ============================================================================


import requests

response = requests.get("https://jsonplaceholder.typicode.com/posts")
posts = response.json()
print("帖子总数：", len(posts))

print("\nuserId=1的帖子：")
for post in posts:
    if post["userId"] == 1:
        print(post["id"], post["title"])

print("\n前5个帖子标题：")
for i in range(5):
    print(posts[i]["title"])

max_post = posts[0]
for post in posts:
    if len(post["title"]) > len(max_post["title"]):
        max_post = post

print("\n标题最长的帖子：")
print("标题：", max_post["title"])
print("ID：", max_post["id"])

# 作业三：POST请求 - 创建数据
# ============================================================================
#
# 任务描述：
# 编写一个程序，使用POST请求创建新数据：
# 1. 使用POST请求创建一个新帖子
#    URL: https://jsonplaceholder.typicode.com/posts
# 2. 请求体包含：title、body、userId
# 3. 打印响应状态码和响应数据
# 4. 验证返回的数据是否包含提交的数据
#
# 要求：
# - 使用requests.post()发送POST请求
# - 使用json参数传递数据
# - 处理响应数据
# - 添加适当的注释
#
# ============================================================================

import requests

body = {
    "title": "1111",
    "body": "2222",
    "userId": 1
}
response = requests.post("https://jsonplaceholder.typicode.com/posts", json=body)

print("响应状态码：", response.status_code)
print("响应数据：")
print(response.json())

res_data = response.json()
if res_data["title"] == body["title"]:
    print("\n验证成功")
else:
    print("\n验证失败")


# 作业四：综合应用 - 简单的API客户端
# ============================================================================
#
# 任务描述：
# 编写一个简单的API客户端程序，实现以下功能：
# 1. 显示菜单：
#    - 1. 获取所有用户
#    - 2. 获取指定用户信息（输入用户ID）
#    - 3. 获取所有帖子
#    - 4. 获取指定用户的帖子（输入用户ID）
#    - 5. 退出
# 2. 使用while循环让程序可以重复执行
# 3. 根据用户选择执行相应的API调用
# 4. 格式化显示API返回的数据
# 5. 处理错误情况（网络错误、无效输入等）
#
# 要求：
# - 使用函数组织代码
# - 使用requests库发送HTTP请求
# - 使用try-except处理错误
# - 输出格式清晰美观
# - 添加详细的注释
#
# ============================================================================



import requests

def get_all_users():
    try:
        res = requests.get( "https://jsonplaceholder.typicode.com/users")
        users = res.json()
        for u in users:
            print(f"ID: {u['id']}, 姓名: {u['name']}, 邮箱: {u['email']}")
    except:
        print("wrong")

def get_user_by_id(user_id):
    try:
        res = requests.get(f"https://jsonplaceholder.typicode.com/users/{user_id}")
        if res.status_code != 200:
            print("用户不存在")
            return
        user = res.json()
        print(f"ID: {user['id']}")
        print(f"姓名: {user['name']}")
        print(f"用户名: {user['username']}")
        print(f"邮箱: {user['email']}")
    except:
        print("wrong")

def get_all_posts():
    try:
        res = requests.get( "https://jsonplaceholder.typicode.com/posts")
        posts = res.json()
        print("所有帖子")
        for p in posts[:10]:
            print(f"ID: {p['id']}, 标题: {p['title']}")
    except:
        print("wrong")

def get_posts_by_user(user_id):
    try:
        url = "https://jsonplaceholder.typicode.com/posts"
        res = requests.get(url)
        posts = res.json()
        print(f"用户 {user_id} 的帖子")
        for p in posts:
            if p["userId"] == user_id:
                print(f"帖子ID: {p['id']}, 标题: {p['title']}")
    except:
        print("wrong")

def show_menu():
    print("菜单:")
    print("1. 获取所有用户")
    print("2. 获取指定用户信息")
    print("3. 获取所有帖子")
    print("4. 获取指定用户的帖子")
    print("5. 退出")

while True:
        show_menu()
        choice = input("请输入选择：")
        if choice == "1":
            get_all_users()

            uid = input("请输入用户ID：")
            get_user_by_id(int(uid))
        elif choice == "3":
            get_all_posts()
        elif choice == "4":
            uid = input("请输入用户ID：")
            get_posts_by_user(int(uid))
        elif choice == "5":
            print("谢谢使用！")
            break
        else:
            print("输入无效，请重输！")


# 提交要求
# ============================================================================
#
# 1. 每个作业创建一个独立的.py文件，命名为：作业一.py、作业二.py 等
# 2. 代码要符合Python代码规范（命名、缩进、注释等）
# 3. 确保代码可以正常运行（需要网络连接）
# 4. 完成后可以对照参考答案检查自己的代码
#
# 注意：
# - 如果网络不可用，可以注释掉实际API调用，使用模拟数据演示
# - 某些API可能有访问限制，如果遇到问题可以尝试其他免费API
#
# 祝学习愉快！








