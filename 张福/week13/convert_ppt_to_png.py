#!/usr/bin/env python3
import os
import subprocess
import glob
import json
from pathlib import Path

# 配置
INPUT_DIR = '/home/zhang/文档/learning_ai_ppt'
OUTPUT_ROOT = 'ppts'

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 获取所有 PPT/PPTX 文件
ppt_files = sorted(list(glob.glob(os.path.join(INPUT_DIR, '*.ppt*'))))
if not ppt_files:
    print('⚠️ 未在', INPUT_DIR, '中找到 .ppt 或 .pptx 文件')
    exit(1)

print(f'✅ 找到 {len(ppt_files)} 个 PPT 文件')

# 逐个处理
slide_records = []
for i, ppt_path in enumerate(ppt_files):
    stem = Path(ppt_path).stem
    # 清理目录名（避免空格/特殊字符）
    dir_name = f'{i+1:02d}_{stem.replace(" ", "_").replace("/", "-")}'
    output_dir = Path(OUTPUT_ROOT) / dir_name
    output_dir.mkdir(exist_ok=True)

    # 使用 libreoffice --headless 导出为 PNG
    cmd = [
        'libreoffice', '--headless', '--convert-to', 'png:impress_png_Export',
        '--outdir', str(output_dir),
        ppt_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        # 重命名：libreoffice 默认输出为 'xxx.png', 'xxx_1.png', ... → 统一为 slide_001.png 等
        pngs = sorted(list(output_dir.glob('*.png')))
        for idx, p in enumerate(pngs, 1):
            new_name = f'slide_{idx:03d}.png'
            p.rename(output_dir / new_name)
        slide_records.append({
            'title': stem,
            'dir': dir_name,
            'count': len(pngs)
        })
        print(f'  → {stem} → {len(pngs)} 张图已存入 {output_dir}')
    except subprocess.CalledProcessError as e:
        print(f'❌ 转换失败 {ppt_path}: {e}')
        continue

# 生成 index.html
html_content = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI 学习 PPT 图库</title>
  <style>
    body { font-family: "Segoe UI", sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 1rem; }
    h1 { text-align: center; color: #2c3e50; }
    .deck { margin-bottom: 2rem; padding: 1rem; border-radius: 8px; background: #f8f9fa; }
    .deck h2 { margin-top: 0; color: #2980b9; }
    .slides { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 0.5rem; }
    .slide-img { width: 100%; height: 120px; object-fit: cover; border: 1px solid #ddd; border-radius: 4px; cursor: pointer; }
    .slide-img:hover { opacity: 0.9; }
    .slide-count { font-size: 0.85em; color: #7f8c8d; }
  </style>
</head>
<body>
  <h1>📚 AI 学习 PPT 图库</h1>
'''

for deck in slide_records:
    html_content += f'  <div class="deck">
    <h2>{deck["title"]} <span class="slide-count">({deck["count"]}页)</span></h2>
    <div class="slides">
'
    for i in range(1, deck['count'] + 1):
        img_path = f'{deck["dir"]}/slide_{i:03d}.png'
        html_content += f'      <img src="{img_path}" alt="{deck["title"]} 第{i}页" class="slide-img" onclick="window.open('{img_path}', '_blank')">
'
    html_content += '    </div>
  </div>
'

html_content += '''
</body>
</html>'''

with open(os.path.join(OUTPUT_ROOT, 'index.html'), 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f'✅ 已生成图库页面：ppts/index.html')
print('💡 提示：确保已安装 LibreOffice CLI（如未安装，请运行 `sudo apt install libreoffice`）')
