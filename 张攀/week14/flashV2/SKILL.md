---
name: "flashcardV2"
description: "Builds an English vocabulary flashcard: an LLM generates word JSON (Chinese meaning, synonyms, near-synonyms, 3 examples), then a Python script renders it to a flip-card HTML. Invoke when user asks to create a flashcard/word-card/vocabulary card, or to look up a word's Chinese meaning, synonyms, and example sentences."
triggers: 闪卡, flashcard, 单词卡片, 单词卡, 单词
entry: script/flashcard.py
argtype: word
---

# 单词闪卡 (FlashcardV2)

两阶段分工：
1. **生成 JSON**：由 harness 所调用的 LLM（agent）按下方 Schema 生成单词数据（不写入本 skill）
2. **渲染 HTML**：`flashcard.run(word=..., data=...)` 读取传入的 JSON，渲染为独立翻转卡片 `data/<word>.html`

## 何时调用

- 用户要做"闪卡 / flashcard / 单词卡片 / 单词卡片 HTML"
- 用户要查询单词的"中文释义 + 同义词 + 近义词 + 例句"并生成卡片
- 用户要批量制作单词学习卡

## 资产清单

| 文件 | 用途 |
|---|---|
| `script/flashcard.py` | 唯一入口模块，导出 `run(word="", data=None, json_path=None) -> dict`：仅渲染 HTML，不调用 LLM |
| `data/<word>.json` | 单词数据（由 harness 侧 LLM 生成，可存于此作为输入） |
| `data/<word>.html` | 渲染输出的翻转卡片 HTML |

> 接口规范：每个 skill 的 `script/` 下只导出一个 `run(**kwargs) -> dict`，由 harness 用 importlib 动态加载后调用 `module.run(**params)`。依赖仅 Python 标准库（`html`/`json`/`re`），无需 pip install。

## JSON 数据来源（由 harness 所调用的 LLM 生成）

flashcard 内部不调用任何 LLM。单词 JSON 由调用方（harness 所调用的 LLM / agent）按下方 Schema 生成：

- 经 harness 传入：`python harness.py "制作 brave 闪卡" --json <brave.json 路径>`，harness 读取文件后作为 `data` 参数调用 `flashcard.run()`
- 或直接调用：`flashcard.run(word="brave", data={...})` / `flashcard.run(word="brave", json_path="...")`

---

## JSON Schema

`data/<word>.json` 必须符合以下结构（字段名用下划线风格，脚本也兼容驼峰）：

```json
{
  "word": "brave",
  "phonetic": "/breɪv/",
  "audio": "",
  "translation": "勇敢的",
  "definitions": [
    {
      "partOfSpeech": "adj",
      "definition": "Ready to face and endure danger or pain; showing courage.",
      "definition_zh": "准备面对并忍受危险或痛苦；表现出勇气。"
    }
  ],
  "synonyms": ["courageous", "valiant", "fearless", "bold", "intrepid"],
  "near_synonyms": ["heroic", "daring", "audacious", "gallant", "stalwart"],
  "examples": [
    { "en": "He was a brave soldier.", "zh": "他是一名勇敢的士兵。" }
  ]
}
```

### 字段要求

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `word` | string | 是 | 英文单词 |
| `phonetic` | string | 否 | 音标，如 `/breɪv/` |
| `audio` | string | 否 | 发音音频 URL，空字符串则不显示按钮 |
| `translation` | string | 是 | 中文释义（核心） |
| `definitions` | array | 否 | 详细释义，每项含 `partOfSpeech`/`definition`/`definition_zh` |
| `synonyms` | array | 是 | 同义词，建议 5 个 |
| `near_synonyms` | array | 是 | 近义词，建议 5 个 |
| `examples` | array | 是 | 例句，建议 3 条，每项含 `en`/`zh` |

脚本会自动截断：同义词/近义词取前 5，例句取前 3。

---

## 供 harness 侧 LLM 生成 JSON 的 Prompt

调用方 LLM（agent）生成单词 JSON 时使用以下 Prompt（让 LLM 严格输出 JSON）：

```
你是一个英语词典助手。请为英文单词 "{word}" 生成闪卡数据，严格按以下 JSON 格式返回，不要任何额外文字：

{
  "word": "单词",
  "phonetic": "音标，如 /breɪv/",
  "audio": "",
  "translation": "最常用的中文释义",
  "definitions": [
    {"partOfSpeech": "词性", "definition": "英文释义", "definition_zh": "中文释义"}
  ],
  "synonyms": ["同义词1", "同义词2", "同义词3", "同义词4", "同义词5"],
  "near_synonyms": ["近义词1", "近义词2", "近义词3", "近义词4", "近义词5"],
  "examples": [
    {"en": "英文例句1", "zh": "中文翻译1"},
    {"en": "英文例句2", "zh": "中文翻译2"},
    {"en": "英文例句3", "zh": "中文翻译3"}
  ]
}

要求：
1. definitions 提供 2-3 条不同词性的释义
2. synonyms 和 near_synonyms 各 5 个，不要重复
3. examples 3 条，例句要自然、难度适中，中文翻译准确
4. 只返回 JSON，不要 markdown 代码块标记，不要解释
```

---

## ReAct 流程

### Step 1 · 确认单词

- **Thought**：需知道为哪个单词做闪卡。
- **Action**：直接取用户输入中的英文单词（如"制作 brave 闪卡"→ brave）；若用户未说明，用 `AskUserQuestion` 询问。
- **Observation**：得到目标单词。

### Step 2 · LLM 生成 JSON 并经 harness 传入

- **Thought**：单词 JSON 由本 LLM（agent）生成，flashcard 只负责渲染。
- **Action**：
  1. 按上方"供 harness 侧 LLM 生成 JSON 的 Prompt"自行生成 `<word>` 的 JSON（符合下方 Schema），用 `Write` 保存到临时文件或 `data/<word>.json`
  2. `RunCommand` 运行 harness：
     ```
     python harness.py "制作 <word> 闪卡" --json <word.json 路径>
     ```
     harness 全量加载本 SKILL.md，读取 JSON 并调用 `flashcard.run(word=..., data=...)`，完成渲染
- **Observation**：返回 dict，`success: true` 且含 `html_path`。
- **异常处理**：若返回 `error` 提到缺字段，说明生成的 JSON 不符合 Schema，修正后重试。

### Step 3 · 校验 JSON

- **Thought**：确保 JSON 符合 schema，否则渲染会出错。
- **Action**：`Read` 读取 `data/<word>.json`，逐项核对：
  - `word` 非空
  - `translation` 非空
  - `synonyms` / `near_synonyms` 各有 5 个
  - `examples` 有 3 条，每条 `en`/`zh` 非空
  - JSON 语法合法（无尾逗号、无注释）
- **Observation**：字段完整则进入 Step 4；缺失则重新执行 Step 2（`run()` 内部已做基本校验）。

### Step 4 · 确认渲染产物

- **Thought**：Step 2 已同时完成渲染。
- **Action**：检查返回 dict 中的 `html_path`，`Read` 确认 `data/<word>.html` 存在且非空。
- **Observation**：HTML 文件就绪，进入浏览器验证。

### Step 5 · 浏览器验证

- **Thought**：确认 HTML 渲染与翻转动画正常。
- **Action**：用 `browser_use` 子代理或 `integrated_browser` MCP 打开生成的 HTML 文件（`file:///` 协议路径），截图正面；触发翻转（点击 `#card` 或 `browser_evaluate` 执行 `document.getElementById('card').classList.toggle('flipped')`），等待 1200ms 后截图背面。
- **Observation**：
  - 正面：单词、音标、中文释义、同义词快览正常，中文无乱码
  - 背面：详细释义（中英对照）、同义词、近义词、3 条例句（中英对照）可见，目标词高亮
- **关键**：翻转截图须在 0.7s 过渡动画完成后抓取（延时 ≥1200ms），否则误判失败。可用 `browser_evaluate` 读 `getComputedStyle(card).transform`，从 `none` 变为 `matrix3d(...)` 即证明翻转生效。

### Step 6 · 交付

- **Thought**：验证通过，交付 HTML 文件。
- **Action**：告知用户 HTML 文件路径，可双击直接在浏览器打开。
- **Observation**：用户获得独立可分享的闪卡 HTML。

---

## 脚本用法速查

```bash
# 经 harness（JSON 由 harness 侧 LLM 先生成到文件，再经 --json 传入）
python harness.py "制作 brave 闪卡" --json .trae/skills/flashcardV2/data/brave.json

# 直接调用模块（CWD 为 skill 目录）
python -c "import importlib.util,json; s=importlib.util.spec_from_file_location('m','script/flashcard.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(json.dumps(m.run(word='brave', json_path='data/brave.json'), ensure_ascii=False, indent=2))"
```

特性：
- `run()` 只负责渲染：从 `data`（dict）或 `json_path`（文件）读取 JSON → 校验 Schema → 输出 HTML
- 自动 HTML 转义防 XSS
- 例句中目标词自动高亮（`<mark>`）
- 同义词/近义词自动截断到 5 个，例句截断到 3 条
- 兼容下划线与驼峰命名字段（`near_synonyms` / `nearSynonyms` 均可）

## 测试用例

- `data/brave.json`：已附示例数据，`python harness.py "制作 brave 闪卡" --json .trae/skills/flashcardV2/data/brave.json` 直接验证渲染
- 生僻词（如 `serendipity`）：验证 harness 侧 LLM 生成的 JSON 渲染质量
- 多词性词（如 `light`）：验证 definitions 多条展示
