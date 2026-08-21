---
name: poem-card
description: >-
  为一首古诗生成静态 HTML 学习卡（原文、注释、译文、赏析、背诵自测）。
  Use when the user asks for 古诗卡 / 诗词学习卡 / poem card，
  例句："做一张《静夜思》的古诗卡"、"《登鹳雀楼》的学习卡"、"生成 poem card：悯农"、"复习《泊船瓜洲》，做个诗词卡"
---

# Poem Card 古诗学习卡

流程：识别诗题 → 写 `<skill_dir>/data/<title>.json` → 跑脚本生成 `./<title>.html` → `open_file` 预览。

工具（优先用，勿试 shell）：
- `write_text_file`：写 `<skill_dir>/data/<title>.json`
- `run_local_script`：执行 `<skill_dir>/scripts/make_poem_card.py`，json 路径和 `-o <title>.html` 放 args
- `open_file`：打开生成的 html

## 流程
1. **识别诗作**：取诗题（去书名号作文件名）；只给作者或描述模糊 → 先向用户确认，不猜。
2. **写 JSON**（契约见下）到 `<skill_dir>/data/<title>.json`。原文必须逐字准确，不确定就提醒核对，不凭印象杜撰。未指明学段 → 按小学中高年级默认难度。
3. **运行脚本**：`script_path` 填 `<skill_dir>/scripts/make_poem_card.py`，args 放 json 路径 + `-o ./<title>.html`，输出到当前工作目录。
4. **打开预览**：`open_file` 打开生成的 HTML。

## 数据契约
```json
{"title":"静夜思","author":"李白","dynasty":"唐","grade_hint":"二年级","lines":["床前明月光，","疑是地上霜。","举头望明月，","低头思故乡。"],"notes":[{"word":"疑","meaning":"好像，仿佛"},{"word":"举头","meaning":"抬起头"},{"word":"思","meaning":"思念，想念"}],"translation":"床前洒满了明亮的月光，好像铺上了一层白白的霜。抬起头望着天上的月亮，低下头思念起远方的家乡。","appreciation":"这首诗用眼前最常见的月光，写出了出门在外的人对家乡的想念。语言简单，感情真挚，是流传最广的思乡诗。","quiz":[{"q":"床前明月＿","a":"光"},{"q":"疑是地上＿","a":"霜"},{"q":"低头思＿","a":"故乡"}]}
```
字段：`title`诗题(去书名号) / `author`作者 / `dynasty`朝代 / `grade_hint`学段(用户没提→空串) / `lines`逐句数组(标点留句尾，一句一元素) / `notes`重点字词注释3-6条{word,meaning} / `translation`整段白话(勿逐句对照) / `appreciation`赏析2-4句 / `quiz`背诵自测3-5条{q挖空句用＿代字，a答案}。数量约束必须满足。
难度：二年级→短诗、大白话、quiz 只挖句尾 1-2 字；初一→可讲意象/表现手法、挖关键字。

## 常见错误
1. 整诗塞成一个字符串 → 脚本按数组逐行排版。
2. 文件名带《》/标点/空格 → 只用诗题文字。
3. quiz 挖空过多 → 每条 1-2 字。
4. 译文写成逐句对照 → 要一整段连贯白话。
