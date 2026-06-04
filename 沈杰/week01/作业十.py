# 第十部分-算法示例 作业说明
# 
# 本部分作业共4道题，从简单到复杂，帮助巩固以下知识点：
# - 使用requests库发送HTTP请求
# - 使用BeautifulSoup解析网页HTML
# - 提取网页中的结构化数据
# - 处理文件读写（JSON格式）
# - 异常处理在网络爬虫中的应用
#
# 注意：本部分作业需要使用以下库：
# - requests：pip install requests
# - beautifulsoup4：pip install beautifulsoup4
# - 本部分作业可以使用之前学过的所有知识
#
# ============================================================================
# 作业一：基础爬虫 - 获取网页内容
# ============================================================================
#
# 任务描述：
# 编写一个程序，爬取正北方网（https://www.northnews.cn/）的首页内容：
# 1. 使用requests库发送GET请求获取网页
# 2. 设置合适的请求头（User-Agent）
# 3. 打印响应状态码
# 4. 获取网页的标题（title标签）
# 5. 打印网页的部分内容（前500个字符）
# 6. 使用异常处理处理网络错误
#
# 要求：
# - 使用requests.get()发送请求
# - 设置合适的请求头
# - 使用try-except处理网络异常
# - 添加适当的注释
#
# ============================================================================

from bs4 import BeautifulSoup
import requests
try:
    response = requests.get("https://www.northnews.cn",headers={},timeout=5)
    print(f'状态码是{response.status_code}')

    response.encoding = response.apparent_encoding
    text = response.text
    soup = BeautifulSoup(text, 'html.parser')
    title = soup.find('title')
    print(f"标题：{title.text}")
    print(f'前两百个字符为:{text[:200]}')
except requests.exceptions.Timeout:
    print(f"请求超时")
except requests.exceptions.ConnectionError:
     print(f"连接错误")
except requests.exceptions.HTTPError as e:
    print(f"HTTP错误：{e}")
except Exception as e:
        print(f"发生错误：{e}")

# 作业二：解析网页 - 提取新闻标题和链接
# ============================================================================
#
# 任务描述：
# 编写一个程序，从正北方网首页提取新闻标题和链接：
# 1. 获取首页HTML内容
# 2. 使用BeautifulSoup解析HTML
# 3. 找到所有的新闻标题和对应的链接
# 4. 提取并显示前10条新闻的标题和链接
# 5. 将提取的数据保存到JSON文件
#
# 要求：
# - 使用BeautifulSoup解析HTML
# - 查找包含新闻标题的元素（可能需要观察网页结构）
# - 提取链接时需要处理相对路径和绝对路径
# - 将数据保存为JSON格式
# - 添加适当的注释
#
# ============================================================================


from bs4 import BeautifulSoup
import requests
import json

try:
    response = requests.get("https://www.northnews.cn",headers={},timeout=5)
    print(f'状态码是{response.status_code}')

    response.encoding = response.apparent_encoding
    text = response.text
    soup = BeautifulSoup(text, 'html.parser')
    news_list = []
    for a in soup.find_all('a', href=True):
         title = a.get_text(strip=True) 
         link = a["href"] 
         if not title or len(title) < 4:
            continue
         news_list.append({
            "title": title,
            "link": link
        })
    top10_news = news_list[:10]
    print("正北方网前10条新闻：\n")
    for i, news in enumerate(top10_news, 1):
        print(f"{i}. {news['title']}")
        print(f"   链接：{news['link']}\n")
    save_path = r"D:\BaiduNetdiskDownload\ai课程作业\第十部分-算法示例\课件及作业\作业\news.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(top10_news, f, ensure_ascii=False, indent=4)   
except requests.exceptions.Timeout:
    print(f"请求超时")
except requests.exceptions.ConnectionError:
     print(f"连接错误")
except requests.exceptions.HTTPError as e:
    print(f"HTTP错误：{e}")
except Exception as e:
        print(f"发生错误：{e}")


# 作业三：分类爬取 - 爬取不同分类的新闻
# ============================================================================
#
# 任务描述：
# 编写一个程序，爬取正北方网不同分类的新闻：
# 1. 定义多个分类的URL（如：内蒙古、国内、国际等）
# 2. 为每个分类创建一个爬取函数
# 3. 从每个分类页面提取新闻列表（标题、链接、时间等）
# 4. 将不同分类的新闻分别保存到不同的JSON文件
# 5. 统计每个分类爬取的新闻数量
#
# 要求：
# - 使用函数组织代码
# - 处理每个分类的网页结构差异
# - 添加适当的延时，避免请求过快
# - 使用异常处理确保某个分类失败不影响其他分类
# - 添加详细的注释
#
# ============================================================================

import requests
from bs4 import BeautifulSoup
import json
import time

CATEGORY_URLS = {
    '内蒙古': 'https://www.northnews.cn/news/neimenggu/',
    '国内': 'https://www.northnews.cn/news/guonei/',
    '国际': 'https://www.northnews.cn/news/guoji/',
}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
SAVE_DIR = r"D:\BaiduNetdiskDownload\ai课程作业\第十部分-算法示例\课件及作业\作业"

def crawl_news(category_name, url):
    """
    爬取单个分类的新闻
   （内蒙古/国内/国际）
    """
    news_list = []
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.encoding = response.apparent_encoding
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        for a_tag in soup.find_all("a", href=True):
            title = a_tag.get_text(strip=True)
            link = a_tag["href"]

            if not title or len(title) < 4:
                continue
            # 只保留新闻链接
            if "html" in link or "news" in link:
                news_list.append({
                    "title": title,
                    "link": link,
                    "category": category_name
                })
    except Exception as e:
        print(f" {category_name} 爬取失败：{str(e)}")
    return news_list

def save_to_json(category_name, news_list):
    filename = f"{SAVE_DIR}\\{category_name}_news.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(news_list, f, ensure_ascii=False, indent=4)
    print(f"已保存到：{filename}")

total_count = 0

for category, url in CATEGORY_URLS.items():
    news = crawl_news(category, url)
    print(f"分类{category}爬取到的数量{len(news)}")
    save_to_json(category, news)

    total_count += len(news)
    time.sleep(2)  

print("\n" + "-" * 50)
print(f"爬取统计：")
print(f"总新闻数：{total_count}")
print(" 全部任务完成！")

# 作业四：综合应用 - 完整的新闻爬虫系统
# ============================================================================
#
# 任务描述：
# 编写一个完整的新闻爬虫系统，参考教学代码的NewsCrawler类：
# 1. 定义一个NorthNewsCrawler类
#    - __init__方法：初始化，设置基础URL和请求头
#    - get_category_urls方法：返回各个分类的URL字典
#    - get_news_list方法：从分类页面获取新闻列表
#    - get_news_content方法：获取单篇新闻的详细内容
#    - crawl_category方法：爬取某个分类的所有新闻
#    - crawl_all_categories方法：爬取所有分类的新闻
#    - save_data方法：将数据保存到JSON文件
# 2. 在主程序中：
#    - 创建爬虫实例
#    - 爬取指定分类或所有分类的新闻
#    - 保存数据y
#    - 显示爬取统计信息
#
# 要求：
# - 使用类组织代码结构
# - 使用Session复用连接，提高效率
# - 添加适当的延时和异常处理
# - 清理提取的文本内容（去除空白、换行等）
# - 支持保存到JSON文件
# - 添加详细的注释和错误处理
#
# ============================================================================

import requests
from bs4 import BeautifulSoup
import json
import time
from requests.exceptions import RequestException

class NorthNewsCrawler:
    def __init__(self):
        self.base_url = "https://www.northnews.cn/"
        self.session = requests.Session()  
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.save_path = r"D:\BaiduNetdiskDownload\ai课程作业\第十部分-算法示例\课件及作业\作业"

    def get_category_urls(self):
        """返回各个新闻分类URL字典"""
        return {
            "首页": "https://www.northnews.cn/",
            '内蒙古': 'https://www.northnews.cn/news/neimenggu/',
            '国内': 'https://www.northnews.cn/news/guonei/',
            '国际': 'https://www.northnews.cn/news/guoji/',
        }

    def clean_text(self, text):
        """清理文本：去除空白、换行、多余空格"""
        if not text:
            return ""
        return " ".join(text.strip().split())

    def get_news_list(self, url):
        """从分类页面获取新闻列表（标题+链接）"""
        news_list = []
        try:
            resp = self.session.get(url, headers=self.headers, timeout=10)
            resp.encoding = "utf-8"
            soup = BeautifulSoup(resp.text, "html.parser")

            for a in soup.find_all("a", href=True):
                title = self.clean_text(a.get_text())
                link = a["href"]

                if len(title) < 4:
                    continue

                if link.startswith("/"):
                    link = self.base_url + link

                if "html" in link or "news" in link:
                    news_list.append({"title": title, "link": link})

        except RequestException as e:
            print(f"获取新闻列表失败：{e}")
        print(f'从分类页面获取新闻列表:{news_list}')
        return news_list

    def get_news_content(self, url):
        """获取单篇新闻详细内容（正文）"""
        try:
            resp = self.session.get(url, headers=self.headers, timeout=10)
            resp.encoding = "utf-8"
            soup = BeautifulSoup(resp.text, "html.parser")
            paragraphs = soup.find_all("p")
            content = "\n".join([self.clean_text(p.get_text()) for p in paragraphs])
            return content[:1000]  # 限制长度
        except:
            print('获取正文失败')
            return "获取正文失败"

    def crawl_category(self, category_name):
        """爬取单个分类所有新闻"""
        print(f"\n正在爬取分类：{category_name}")
        categories = self.get_category_urls()
        if category_name not in categories:
            print("分类不存在！")
            return []

        url = categories[category_name]
        news_list = self.get_news_list(url)

        # 为每条新闻补充正文
        for news in news_list[:10]:
            news["content"] = self.get_news_content(news["link"])
            news["category"] = category_name
            time.sleep(0.1)

        print(f"分类 {category_name} 爬取完成，共 {len(news_list)} 条")
        return news_list

    def save_data(self, data, category_name):
        """保存数据到JSON文件"""
        filename = f"{self.save_path}\\{category_name}_news.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"已保存到：{filename}")

    def crawl_all_categories(self):
        """爬取所有分类"""
        all_stats = {}
        categories = self.get_category_urls()

        for name in categories:
            print("crawl_category start")
            news = self.crawl_category(name)
            print("save_data start")
            self.save_data(news, name)
            all_stats[name] = len(news)
            time.sleep(2)

        return all_stats

crawler = NorthNewsCrawler()
stats = crawler.crawl_all_categories()
print("\n===== 爬取统计 =====")
total = 0
for cate, count in stats.items():
    print(f"{cate}：{count} 条")
    total += count
print(f"总计：{total} 条新闻")
print("爬虫end")

# 提交要求
# ============================================================================
#
# 1. 每个作业创建一个独立的.py文件，命名为：作业一.py、作业二.py 等
# 2. 代码要符合Python代码规范（命名、缩进、注释等）
# 3. 确保代码可以正常运行（需要网络连接）
# 4. 完成后可以对照参考答案检查自己的代码
#
# 注意：
# - 爬虫需要遵守法律法规和网站规定，仅用于学习目的
# - 添加适当的延时，避免对服务器造成压力
# - 如果网站结构发生变化，可能需要调整代码
# - 某些网站可能有反爬虫机制，如遇问题可以降低爬取频率
#
# 祝学习愉快！
