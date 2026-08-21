---
name: ppt_to_html
description: |
  ## PPT 拆解工具

  当用户上传了 .ppt/.pptx 文件并要求拆解为逐页图片和 HTML 浏览页面时，你必须调用 `ppt_to_html` 函数。

  ### 调用方式

  函数名：`ppt_to_html`

  参数：
  | 参数 | 类型 | 必填 | 说明 |
  |------|------|------|------|
  | `ppt_files` | array[string] | 是 | 要处理的 PPT/PPTX 文件绝对路径列表 |
  | `session_id` | string | 否 | 当前会话 ID，用于生成唯一输出目录 |

  示例调用：
  ```json
  {"ppt_files": ["/home/zhang/ai_learning/week13_harness/files/课件1.pptx"], "session_id": "abc123"}
  ```

  ### 使用说明
  - 用户上传 PPT 文件后，消息中会包含文件本地路径信息和会话 ID
  - 你必须提取这些路径并传入 `ppt_files` 参数，同时传入 `session_id`
  - 输出目录自动为 `files/ppts/{session_id}_时间戳/`，确保每次拆解结果独立不覆盖
  - 告诉用户 HTML 文件的最终访问路径

  ### 适用场景
  - "帮我把上传的 PPT 拆解成 HTML"
  - "把刚才上传的课件转为图片和网页"
  - "处理我上传的 PPT 文件"
---

## 实现步骤

```
用户请求 "将我上传的 PPT 拆解为 HTML"
  │
  ├─ 1. ppt_to_html(ppt_files, session_id="")
  │      入口函数，接收文件路径列表和会话 ID
  │      │
  │      ├─ 1.1 参数校验
  │      │      验证 ppt_files 非空
  │      │      → 每个路径展开 resolve()，检查存在
  │      │      → 检查后缀 .ppt / .pptx
  │      │
  │      ├─ 1.2 输出目录
  │      │      使用 session_id + 时间戳 生成唯一目录名
  │      │      files/ppts/{session_id}_YYYYMMDD_HHMMSS/
  │      │      → 已存在则清空重建
  │      │
  │      ├─ 1.3 逐文件处理
  │      │      for pptx in valid_files:
  │      │      │
  │      │      ├─ 创建子目录 _sanitize(pptx.stem)
  │      │      │
  │      │      ├─ Presentation(str(pptx))  # python-pptx 打开
  │      │      │
  │      │      └─ for i, slide in enumerate(prs.slides):
  │      │              │
  │      │              ├─ _render_slide(slide, i, pptx.stem)
  │      │              │     → 返回 PIL Image
  │      │              │
  │      │              └─ img.save(sub / f"slide_{i+1:03d}.png")
  │      │                    → 记录 (dir_name, filename)
  │      │
  │      └─ 1.4 HTML 生成
  │             _build_html(all_slides, ppt_names)
  │             → 写入 output_dir/index.html
  │
  └─ 返回 {success, message, output_dir, html_file, total_ppts, total_slides}
```

## 内部方法细则

### `ppt_to_html(ppt_files, session_id="")` — 入口函数

**调用时机**：技能执行器发现 `tool_name == "ppt_to_html"` 时调用。

**参数**：
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `ppt_files` | `List[str]` | 是 | 待处理的 PPT/PPTX 文件绝对路径列表 |
| `session_id` | `str` | 否 | 当前会话 ID，用于生成唯一输出目录 |

**返回值** `Dict[str, Any]`：
| 字段 | 类型 | 说明 |
|------|------|------|
| `success` | `bool` | 是否成功 |
| `message` | `str` | 概要信息 |
| `output_dir` | `str` | 输出目录绝对路径 |
| `html_file` | `str` | HTML 文件绝对路径 |
| `total_ppts` | `int` | 处理的课件数 |
| `total_slides` | `int` | 总页数 |
| `error` | `str` | 失败时返回错误描述 |

**内部流程**：
1. 校验 `ppt_files` 非空
2. 输出目录：若指定 `output_dir` 则使用；否则用 `{session_id}_{时间戳}` 生成唯一目录 `files/ppts/{tag}/`
3. 存在则 `shutil.rmtree(dst)` 清空，再 `mkdir(parents=True)`
4. 逐路径展开并校验存在性和后缀 `.pptx` / `.ppt`
5. 遍历每个有效 PPT 文件：
   - 调用 `_sanitize(stem)` 生成子目录名
   - `Presentation(str(pptx))` 打开
   - 遍历每页调用 `_render_slide` 渲染 → 保存 PNG
6. 调用 `_build_html` 生成 HTML 字符串 → 写入 `index.html`
7. 返回结果字典

---

### `_num_key(name)` — 自然序排序键

**用途**：使 `1_`, `2_`, ..., `9_`, `10_` 按数值升序排列。

**实现**：
```python
m = re.match(r"(\d+)", name)
return int(m.group(1)) if m else 9999
```

**调用者**：不再使用（保留了实现，但 `ppt_to_html` 直接接收路径列表，无需排序）
**参数**：文件名 `str`
**返回**：`int` — 匹配到的首个数字，未匹配则返回 `9999`（排到末尾）

---

### `_sanitize(name)` — 文件名清理

**用途**：将 PPT 文件名转为安全的目录名。

**实现**：
```python
name = re.sub(r"[^\w\u4e00-\u9fff\-]", "_", name)
return name.strip("_") or "ppt"
```

**规则**：
- 只保留汉字、字母、数字、下划线、短横
- 其余字符替换为 `_`
- 首尾 `_` 去掉
- 若结果为空则用 `"ppt"` 兜底

**调用者**：`ppt_to_html` 中生成子目录名
**参数**：PPT 文件名（不含后缀）
**返回**：`str` — 安全的目录名

---

### `_font(size)` — 字体查找

**用途**：按优先级查找系统中文字体，fallback 到 Pillow 默认字体。

**调用链**：`_render_slide` → `_font(48)` / `_font(36)` / `_font(28)` / `_font(22)`

**查找优先级**：
1. `/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc`
2. `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`
3. `/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc`
4. `/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc`
5. `/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf`
6. `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`

**参数**：`size: int` — 字号（px）
**返回**：`ImageFont.FreeTypeFont` 或 `ImageFont.Default`

---

### `_render_slide(slide, slide_idx, ppt_name)` — PPT 页面渲染

**用途**：将 python-pptx 的 Slide 对象渲染为 1920×1080 的 PIL Image。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `slide` | `pptx.slide.Slide` | python-pptx 的 Slide 对象 |
| `slide_idx` | `int` | 页码（0-based） |
| `ppt_name` | `str` | 课件名（标题备用） |

**返回**：`PIL.Image.Image` — RGB 模式，1920×1080

**渲染流程**：

```
开始
│
├─ 创建画布 Image.new("RGB", (1920,1080), "#ffffff")
│  ImageDraw.Draw(img)
│
├─ 读取 4 个字号字体
│  tf=48(标题), hf=36(小标题), bf=28(正文), sf=22(页码)
│
├─ 遍历 slide.shapes:
│  │
│  ├─ 文本提取（shape.has_text_frame）
│  │  for para in shape.text_frame.paragraphs:
│  │  │  para.text.strip() → 跳过空文本
│  │  │  └─ 分类存入:
│  │  │     title → shape.name 含 "title"
│  │  │     heading → para.level==0 且 len<60
│  │  │     body → 其余
│  │
│  └─ 图片提取（shape.shape_type == 13）
│     shape.image.blob → Image.open → thumbnail(400,300)
│     → 粘贴到右侧区域 (left=1056, top=y+10)
│
├─ 绘制标题区
│  │  有标题 → 蓝色背景条 + 标题文字, y=110
│  └─ 无标题 → "课件名 - 第 N 页", y=100
│
├─ 绘制正文区（自动换行）
│  for each body text:
│  │  heading → "▪ text" 36px, y+=50
│  └─ body → textbbox 计算折行, 每行 28px, y+=38
│     → 超过 SLIDE_H-60 截断，最多 8 行
│
└─ 绘制页码
   "N" 居中底部, y=SLIDE_H-40
   分割线 y=SLIDE_H-55
```

**换行算法**（正文 `body` 级别）：
```python
lines = []
cur = ""
for c in text:
    if draw.textbbox((0,0), cur+c, font=bf)[2] > SLIDE_W - 120:
        lines.append(cur)   # 超出宽度 → 换行
        cur = c
    else:
        cur += c
if cur: lines.append(cur)
# 最多取前 8 行绘制
```

---

### `_build_html(slides, ppt_dirs)` — HTML 页面生成

**用途**：将幻灯片列表构建为完整的单页 HTML。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `slides` | `List[tuple]` | `[(dir_name, filename), ...]` 每个元素标识一张图片 |
| `ppt_dirs` | `list` | 课件名列表（用于统计） |

**返回**：`str` — 完整 HTML 源码

**生成的 HTML 结构**：
```
┌─────────────────────────────────────────────┐
│  .layout (flex)                              │
│  ┌──────────────┬──────────────────────────┐ │
│  │ .sidebar     │ .main                    │ │
│  │ 320px        │  flex:1 (图片展示)        │ │
│  │              │                          │ │
│  │ 标题区域      │  <img src='...'>         │ │
│  │ (sticky top)  │  或 placeholder 提示     │ │
│  │              │                          │ │
│  │ 分组 1 (展开) │                          │ │
│  │  第1页       │                          │ │
│  │  第2页       │                          │ │
│  │  ...         │                          │ │
│  │              │                          │ │
│  │ 分组 2 (折叠) │                          │ │
│  │ 分组 3 (折叠) │                          │ │
│  └──────────────┴──────────────────────────┘ │
│  .nav-hint (fixed 右下)                      │
└─────────────────────────────────────────────┘
```

**交互功能**：
| 功能 | 实现 |
|------|------|
| 点击左侧页码 | `show(idx)` → 更新右侧图片，高亮当前项，自动滚动 |
| 点击分组标题 | `toggleGroup(el)` → 切换 `.collapsed` 类，折叠/展开 |
| 键盘 ↑↓ | `keydown` 监听 → `cur+1` / `cur-1` → 调用 `show()` |
| 排序 | 按传入的 ppt_files 列表顺序 + slide_xxx 文件名序 |

**JavaScript 接口**：
| 函数 | 说明 |
|------|------|
| `show(i)` | 显示第 i 张幻灯片：高亮 → 滚动 → 设置 `<img>` |
| `toggleGroup(el)` | 切换分组的折叠状态（group-title 和 slide-group 同步） |
