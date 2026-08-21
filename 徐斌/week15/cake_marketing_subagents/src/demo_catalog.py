"""
本地蛋糕样例目录：模拟浏览器搜索失败时降级，保证教学链路可跑通。
图片均为可公开访问的示例 URL（Unsplash），仅用于演示。
"""
from __future__ import annotations

DEMO_PRODUCTS = [
    {
        "name": "经典芝士蛋糕",
        "price": "¥128-168/6寸",
        "spec": "6寸/8寸；原味/半熟芝士",
        "desc": "进口奶油奶酪烘烤，口感绵密微酸，适合下午茶与轻生日场景。",
        "image": "https://images.unsplash.com/photo-1533134242810-d4caa30c5c4b?w=800",
        "url": "https://example.com/demo/cheesecake",
        "tags": ["芝士", "cheesecake", "下午茶", "生日"],
    },
    {
        "name": "黑森林蛋糕",
        "price": "¥158-218/8寸",
        "spec": "8寸；樱桃酒渍夹心",
        "desc": "巧克力海绵胚配樱桃与淡奶油，节日与生日热销款，视觉层次强。",
        "image": "https://images.unsplash.com/photo-1578985545062-69928b1d9587?w=800",
        "url": "https://example.com/demo/black-forest",
        "tags": ["黑森林", "巧克力", "生日", "节日"],
    },
    {
        "name": "草莓奶油生日蛋糕",
        "price": "¥138-198/6-8寸",
        "spec": "6/8寸；鲜奶油+时令草莓",
        "desc": "新鲜草莓点缀奶油裱花，少女风/亲子生日首选，拍照出片率高。",
        "image": "https://images.unsplash.com/photo-1464349095431-e9a21285b5f3?w=800",
        "url": "https://example.com/demo/strawberry",
        "tags": ["草莓", "奶油", "生日", "水果"],
    },
    {
        "name": "提拉米苏蛋糕",
        "price": "¥148-188/盒装",
        "spec": "方形/圆形；马斯卡彭+咖啡",
        "desc": "意式经典，咖啡酒香与可可粉装饰，适合情侣与白领礼赠。",
        "image": "https://images.unsplash.com/photo-1571877227200-a0d98ea607e9?w=800",
        "url": "https://example.com/demo/tiramisu",
        "tags": ["提拉米苏", "咖啡", "礼赠", "下午茶"],
    },
    {
        "name": "芒果慕斯蛋糕",
        "price": "¥168-228/6寸",
        "spec": "6寸；芒果果泥夹心",
        "desc": "热带果香慕斯，夏季爆款；渐变色层适合社交平台种草。",
        "image": "https://images.unsplash.com/photo-1565958011703-44f9829ba187?w=800",
        "url": "https://example.com/demo/mango-mousse",
        "tags": ["芒果", "慕斯", "夏季", "生日"],
    },
    {
        "name": "抹茶红豆蛋糕",
        "price": "¥148-198/6寸",
        "spec": "6寸；宇治抹茶+蜜红豆",
        "desc": "日式清苦回甘，低糖诉求人群喜爱，包装偏简约和风。",
        "image": "https://images.unsplash.com/photo-1588195538326-c5b1e9f80c5d?w=800",
        "url": "https://example.com/demo/matcha",
        "tags": ["抹茶", "红豆", "日式", "下午茶"],
    },
]


def search_demo_catalog(query: str, max_results: int = 5) -> dict:
    q = (query or "").lower()
    scored = []
    for p in DEMO_PRODUCTS:
        hay = " ".join([p["name"], p["desc"], " ".join(p["tags"])]).lower()
        score = sum(1 for t in p["tags"] if t.lower() in q or t.lower() in hay and t.lower() in q)
        # 宽松匹配：查询词任一字出现在名称/标签
        if any(tok in hay for tok in q.replace("|", " ").split() if len(tok) >= 2):
            score += 2
        if "蛋糕" in q or "cake" in q:
            score += 1
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda x: -x[0])
    picked = [p for _, p in scored[:max_results]] or DEMO_PRODUCTS[:max_results]

    results = [
        {
            "title": f"{p['name']} · {p['price']}",
            "url": p["url"],
            "content": f"{p['desc']} 规格：{p['spec']} 价格：{p['price']}",
        }
        for p in picked
    ]
    images = [p["image"] for p in picked]
    answer = "（演示目录）匹配到：" + "、".join(p["name"] for p in picked)
    return {
        "answer": answer,
        "results": results,
        "images": images,
        "response_time": 0,
        "source": "demo_catalog",
    }
