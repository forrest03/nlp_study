# USAGE_GUIDE.md — 使用与演示指南

## 一、环境准备

```bash
cd 徐斌/week13/progressive_skills_harness
pip install -r requirements.txt
# 若本机已有 conda / 已安装 openai+fastapi，也可直接用该 Python 解释器
```

配置 API Key（与 week12 一致，默认 DeepSeek）：

```bash
export DEEPSEEK_API_KEY="sk-xxx"
# 或
export LLM_PROVIDER=qwen
export DASHSCOPE_API_KEY="sk-xxx"
```

可选：`export AGENT_MODEL=...` 覆盖默认模型名。

---

## 二、CLI 演示

```bash
python src/agent.py
python src/agent.py -q "给我做张 resilient 的闪卡" --once
python src/agent.py -q "统计这段话的字数：渐进式披露节省 context" --once
```

交互命令：

| 输入 | 作用 |
|------|------|
| 普通句子 | 启动一轮 Harness 循环 |
| `/skills` | 打印 L0 索引 |
| `quit` / `q` | 退出 |

观察点：

1. 启动时打印 **L0 索引 tokens** 与 **全量加载 tokens** 对比
2. 出现 `activate_skill(...)` 时 L1 上升
3. 画架构图时若调用 `read_skill_file`，可见 L2
4. 结束时的加载事件列表对应课件「Skill 生命周期」

---

## 三、Web 教学界面

```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8013
```

浏览器打开 `http://localhost:8013`

- 左侧：对话与工具轨迹
- 右侧：当前 / 全量 / 节省 tokens，以及 L0/L1/L2/release 事件
- 产物可通过 `/workspace/...` 访问（如生成的 HTML / SVG）

---

## 四、推荐演示话术（课堂 5 分钟）

1. **先看索引**：打开右侧 `SKILLS.md`，强调「常驻只有摘要」。
2. **跑闪卡**：`给我做张 crazy 的闪卡` → 看到 `activate_skill("flash-card")` → `write_file` → `run_skill_script`。
3. **对比数字**：指出「当前 ≪ 全量」。
4. **跑架构图**：强调只有需要时才 `read_skill_file("baoyu-diagram", "references/architecture.md")`，不会一次加载四个 references。

---

## 五、新增自己的 Skill

1. 创建目录 `skills/my-skill/SKILL.md`
2. Frontmatter 至少包含：

```yaml
---
name: my-skill
description: 一句话说明何时使用
triggers:
  - 关键词1
  - 关键词2
---
```

3. 正文写清步骤，需要脚本时放 `scripts/`，大文档放 `references/` 并在正文里要求「先 activate，再 read_skill_file」。
4. 重启 CLI/Web；`SkillRegistry` 会自动扫描并重写 `SKILLS.md`。

---

## 六、安全说明（教学级）

Harness 对 `run_skill_script` 做了简单黑名单，并对 `write_file` / `read_skill_file` 做了路径约束。这是课件里「工具白名单 / 参数约束」的最小实现，**不适合**直接上生产。生产环境请叠加沙箱、人工审批与审计日志。
