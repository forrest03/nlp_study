---
name: "ppt-generator"
description: "Generates a 16:9 PPT deck from a topic via LLM-generated outline + dynamic template style extraction. Invoke when user asks to create slides/PPT/deck/presentation from a topic, document, or outline."
---

# PPT 生成器（LLM 大纲 + 动态模板排版）

## 调用时机
用户要把主题/文档/要点做成 PPT、幻灯片、演示文稿、slides、deck、presentation 时调用。

## 工作流（LLM 与代码分工）
```
用户给主题
  ↓ [outline_generator.py] LLM 生成 JSON 大纲（调 dashscope）
  ↓ [generate_ppt.py]     动态读取模板配色/字体/尺寸 + 排版成 pptx
生成 pptx
```
LLM 负责理解内容、提炼要点、组织结构；代码负责按模板规范精确排版。两者各司其职。

## 脚本说明

### 1. outline_generator.py — LLM 生成大纲
调 dashscope（`deepseek-v4-flash-0731`）根据主题生成结构化大纲。
- 输入：主题字符串 + 期望页数
- 输出：JSON 大纲 `[{section,title,subtitle,points}, ...]`，首项为封面
- 依赖：`openai`、环境变量 `DASHSCOPE_API_KEY`

```bash
python outline_generator.py "主题" 5                  # 打印 JSON + token 用量到 stderr
python outline_generator.py "主题" 5 > outline.json   # 大纲存文件（token 用量仍到 stderr）
```

### 2. generate_ppt.py — 动态模板排版
把 JSON 大纲排版成 pptx。`--template` 动态套用任意 pptx 模板的配色/字体/尺寸。
- 动态提取：尺寸、字体（theme 主字体）、配色（按字号+位置+亮度推断角色）、页脚文字
- 无模板时用内置默认规范

```bash
python generate_ppt.py outline.json out.pptx                              # 默认规范
python generate_ppt.py outline.json out.pptx --template 模板.pptx          # 套用模板
python generate_ppt.py                                                     # 内置示例
```

## 大纲格式
```json
[
  {"section":"封面","title":"大标题","subtitle":"副标题","points":["亮点1","亮点2","亮点3"]},
  {"section":"01 章节","title":"页标题","subtitle":"页副标题","points":["要点1","要点2","要点3"]}
]
```

## 完整使用步骤
1. 向用户确认 PPT 主题、期望页数、是否有指定模板。若用户已给大纲，跳过第 2 步。
2. `python outline_generator.py "主题" 页数 > outline.json` 生成大纲。
3. `python generate_ppt.py outline.json 输出.pptx --template 模板.pptx` 排版（模板可选）。
4. 把 pptx 路径告知用户，说明可手动微调。

## 依赖
- `openai`（LLM 调用）、`python-pptx` + `lxml`（pptx 生成与模板解析）
- 环境变量 `DASHSCOPE_API_KEY`

## 模板位置
原模板 `自进化agent.pptx` 在 `../../../自进化agent.pptx`（相对本 skill 目录，即 week14 根目录）。也可传入任意 pptx 模板路径。

## 局限
- 配色提取为启发式推断（按字号/位置/亮度），复杂模板可能不完美
- 固定坐标布局，要点超 6 个溢出（需手动分页）
- 仅文本，不支持图片/表格/图表
