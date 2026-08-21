---
name: poem-card
description: >-
  为一首古诗生成静态 HTML 学习卡（原文、注释、译文、赏析、背诵自测填空）。
  Use when the user asks for 古诗卡 / 诗词学习卡 / poem card,
  e.g. "做一张《静夜思》的古诗卡"
---

# Poem Card 古诗学习卡生成

为一首古诗生成一张静态 HTML 学习卡。卡片版面顺序：标题+作者朝代 → 原文逐句 → 注释 → 译文 → 赏析 → 背诵自测（填空，答案点击展开）。

上方系统会注入当前 skill 的目录路径 `skill_dir`。这个 skill 目录下有：
- `data/`：放古诗 json 数据
- `scripts/make_poem_card.py`：根据 json 生成 html

处理这类任务时，优先使用这些工具：
- `write_text_file`：写入 `<skill_dir>/data/<title>.json`
- `run_local_script`：执行 `<skill_dir>/scripts/make_poem_card.py`
- `open_file`：打开生成好的 html

注意：
- 不要把 shell 命令整串塞进 `run_local_script`
- `run_local_script` 的 `script_path` 必须是一个真实存在的脚本文件路径
- 不需要先用 `/bin/bash`、`ls -la`、`/usr/bin/python3 -c` 这类方式试探

## 触发场景

当用户说出类似下面的话时触发本 skill：
- "做一张《静夜思》的古诗卡"
- "《登鹳雀楼》的学习卡"
- "生成 poem card：悯农"
- "复习《泊船瓜洲》，做个诗词卡"

## 执行流程

1. **识别诗作**：从用户话语中提取诗题（去掉书名号作为文件名）。如果只给了作者或描述模糊，先向用户确认具体是哪一首，不要猜。

2. **生成 JSON 数据**：自己写出该诗的学习数据，字段如下，保存到当前 skill 的 `data/` 目录：
   - 路径：`<skill_dir>/data/<title>.json`
   - `title`：诗题（不含书名号）
   - `author`：作者
   - `dynasty`：朝代（如 `唐`）
   - `grade_hint`：适合学段（如 `二年级`、`初中`），用户没提就留空字符串
   - `lines`：原文逐句数组，**一句一个元素，标点保留在句尾**
   - `notes`：重点字词注释数组，每条含 `word` 和 `meaning`（3-6 条）
   - `translation`：全诗白话译文（一整段）
   - `appreciation`：简短赏析（2-4 句，讲清写了什么、好在哪里）
   - `quiz`：背诵自测填空，3-5 条，每条 `q` 为挖空句（用 `＿` 代替要考的字词）、`a` 为答案

   内容要求：
   - 原文必须逐字准确；不确定的诗宁可提醒用户核对，也不要凭印象杜撰
   - 面向二年级：选短诗，注释和译文用大白话，quiz 只挖每句末尾 1-2 个字
   - 面向初一：赏析可以讲意象和表现手法，quiz 可以挖关键字
   - 没有指明孩子时，按默认难度（小学中高年级）生成

3. **生成 HTML**：运行脚本，HTML 输出到**当前工作目录**（不是 skill 目录）：
   - 脚本路径：`<skill_dir>/scripts/make_poem_card.py`
   - 数据路径：`<skill_dir>/data/<title>.json`
   - 推荐输出：`./<title>.html`
   - 调用方式：把脚本路径作为 `run_local_script.script_path`，把 json 路径和 `-o` 输出路径放进 `args`

4. **打开预览**：用 `open_file` 打开生成的 HTML 文件，让用户立即看到效果。

## 数据 JSON 示例

```json
{
  "title": "静夜思",
  "author": "李白",
  "dynasty": "唐",
  "grade_hint": "二年级",
  "lines": ["床前明月光，", "疑是地上霜。", "举头望明月，", "低头思故乡。"],
  "notes": [
    {"word": "疑", "meaning": "好像，仿佛"},
    {"word": "举头", "meaning": "抬起头"},
    {"word": "思", "meaning": "思念，想念"}
  ],
  "translation": "床前洒满了明亮的月光，好像铺上了一层白白的霜。抬起头望着天上的月亮，低下头思念起远方的家乡。",
  "appreciation": "这首诗用眼前最常见的月光，写出了出门在外的人对家乡的想念。语言简单，感情真挚，是流传最广的思乡诗。",
  "quiz": [
    {"q": "床前明月＿", "a": "光"},
    {"q": "疑是地上＿", "a": "霜"},
    {"q": "低头思＿", "a": "故乡"}
  ]
}
```

## 常见错误

1. **把整首诗塞成一个字符串** —— 必须一句一个数组元素，脚本按数组逐行排版。
2. **文件名带书名号** —— `《》`、标点、空格都不要出现在 json 文件名里，只用诗题文字。
3. **quiz 挖空太多** —— 每条只挖 1-2 个字，挖空太多就没法背了。
4. **译文写成逐句对照** —— `translation` 是一整段连贯的白话文，不是逐句翻译列表。
