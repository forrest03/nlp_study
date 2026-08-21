---
name: "stock-dashboard-optimized"
description: "【优化版】暗黑科技风格股票看板。支持公司名+日期查询，拉取30分钟K线，内置代码缓存和快速重试，执行效率更高。Invoke when user asks for stock info/query/dashboard or mentions 股票/K线/行情/看板."
---

# Stock Dashboard Optimized · 股票信息看板（优化版）

## 功能
- 输入「公司名 + 日期」，自动拉取30分钟K线
- 缓存股票代码映射（首次调用后本地缓存，避免重复查询5000+股票列表）
- 快速重试机制（缩短退避时间，减少等待）
- 精简输出（仅返回结构化摘要，降低token消耗）
- 生成JSON数据和独立HTML看板页面

## 参数
| 参数 | 说明 | 示例 |
|------|------|------|
| 公司名 | A股公司名称 | `宁德时代`、`贵州茅台` |
| 日期 | 交易日期 YYYY-MM-DD | `2026-08-04` |

## 调用示例
```bash
python scripts/fetch_stock_opt.py --company 宁德时代 --date 2026-08-04
```

## 渐进式执行步骤（Agent Side）
1. **Step 1 · 执行**：运行 `python scripts/fetch_stock_opt.py --company X --date Y`，自动缓存代码、获取数据、生成JSON+HTML
2. **Step 2 · 查看**：打开 `html_data/{公司}_{日期}.html` 或访问FastAPI服务