# USAGE_GUIDE.md — 企业信息调查 Agent 使用指南

## 1. 环境准备

```bash
cd market_research_subagents
pip install -r requirements.txt
```

在项目根目录创建 `.env`：

```env
DEEPSEEK_API_KEY=sk-xxx
TAVILY_API_KEY=tvly-xxx
```

服务、命令行和模块调用会自动加载该文件；系统中已设置的同名环境变量优先，不会被 `.env`
覆盖。两个密钥都不应写入代码、日志或版本库。

## 2. 发起企业调查

命令行示例：

```bash
python src/agents.py
```

作为 Python 模块调用：

```python
import sys
sys.path.insert(0, "src")

from agents import run_research

report = run_research(
    "调查美团，为上海 Java 后端求职者重点核验业务稳定性、福利和风险"
)
print(report["final_answer"])
```

输入长度为 2–240 个字符。直接输入企业名称即可启动六维并行调查；也可补充意向岗位、城市和特别关注项。仅明确的单一可核验事实（如“华为 2025 年研发人员规模是多少”）会走轻量检索而不派发全部子 Agent。

## 3. Web 服务与可视化

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8002
```

打开 `http://localhost:8002`，输入企业名称和可选求职关注点。页面会显示：

- 主 Agent 的派发决策；
- 六个企业核验节点和各自的 ReAct 过程；
- 子任务并行统计；
- 带来源、信息缺口、风险核验与推荐值的最终报告。

HTTP 接口为 `POST /query`，请求体如下：

```json
{"question": "调查小米集团，为产品经理求职者提供福利与风险参考"}
```

接口以 SSE 依次返回 `start`、`main_step`、`dispatch`、`subagent_step`、`subagent_done`、`final` 与 `done` 事件。`GET /health` 只显示密钥是否就绪，不返回密钥值。

## 4. 如何解读报告

推荐值满分 100，评分构成为经营稳健性 30、业务与成长性 20、薪酬福利 20、雇主体验 15、合规与风险 15。信息缺失应被标为“公开信息未披露”或“无法判断”，而非被模型推测补齐。

报告中的负面信息会区分已生效处罚/判决、进行中的案件和待核验报道。请通过报告的来源 URL 查看原始公告或权威报道，再作求职决定；面试时也应核实薪资结构、社保公积金、试用期、工时、汇报关系和团队稳定性。

## 5. 测试与诊断

不需要联网或 API Key 的核心测试：

```bash
python -m unittest discover -s tests -v
python -m py_compile src/agents.py src/tavily_search.py src/serve.py src/react_loop.py src/llm_client.py
```

如需量化并行与串行的墙钟差异，可执行：

```bash
python src/eval_compare.py --limit 2
```

它会消耗 LLM 与搜索 API 配额，且输出只用于运行时性能对比，不代表企业评分结论。
