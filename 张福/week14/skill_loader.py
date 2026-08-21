# -*- coding: utf-8 -*-
"""
Skill 加载与匹配模块：
1. 加载 pmp_skills/*.md 文件，解析 YAML 元数据与 Markdown 正文；
2. 根据用户提问匹配最相关的 skill（复用 validator.match_standard 的 area 信号）；
3. 回退用 skill.md 的 triggers 触发词匹配；
4. 返回匹配到的 skill 名称和正文内容，供 LLM 上下文注入使用。

兼容性：不修改 validator.py，仅消费其 match_standard/is_pmp_related 返回值。
"""
import os
import re
import threading

import validator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SKILLS_DIR = os.path.join(BASE_DIR, "pmp_skills")

# area 中文 → skill name 映射（从 YAML front matter 中的 name 字段建立）
# 在 _load_skills() 时动态填充
_AREA_TO_SKILL = {}

# 线程锁，保护缓存读写
_LOCK = threading.Lock()


def _parse_yaml_front_matter(text):
    """
    轻量手写 YAML front matter 解析器（不引入 pyyaml 依赖）。
    仅支持简单键值对和列表（- "value" 格式）。
    """
    meta = {}
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return meta

    yaml_text = match.group(1)
    current_key = None

    for line in yaml_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        # 列表项：  - "value"
        if stripped.startswith("- ") and current_key:
            value = stripped[2:].strip().strip('"').strip("'")
            if current_key not in meta:
                meta[current_key] = []
            meta[current_key].append(value)
            continue

        # 键值对：key: value
        kv_match = re.match(r"^(\w+)\s*:\s*(.*)", stripped)
        if kv_match:
            key = kv_match.group(1)
            val = kv_match.group(2).strip().strip('"').strip("'")
            if val:
                # 有值，存为标量
                try:
                    meta[key] = int(val)
                except ValueError:
                    meta[key] = val
            else:
                # 无值，可能是列表的起始
                meta[key] = []
            current_key = key

    return meta


def _parse_skill_md(filepath):
    """
    解析一个 skill.md 文件，返回：
      {
        "name": str,         # skill 名称（英文，如 integration_management）
        "area": str,         # 知识领域中文名（如 项目整体管理）
        "area_en": str,      # 知识领域英文名
        "record_count": int, # 标准问答条数
        "triggers": list,    # 触发词列表
        "process_groups": list,  # 涉及的过程组
        "content": str,      # Markdown 全文（含 front matter + 正文）
        "body": str,         # Markdown 正文（不含 front matter）
      }
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    meta = _parse_yaml_front_matter(text)

    # 提取正文（去掉 front matter）
    body = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, count=1, flags=re.DOTALL)

    return {
        "name": meta.get("name", ""),
        "area": meta.get("area", ""),
        "area_en": meta.get("area_en", ""),
        "record_count": meta.get("record_count", 0),
        "triggers": meta.get("triggers", []),
        "process_groups": meta.get("process_groups", []),
        "content": text,
        "body": body.strip(),
    }


# 模块级缓存
_SKILLS_CACHE = None


def _load_skills():
    """加载全部 skill.md 文件，返回列表，并建立 area→skill 映射。线程安全。"""
    global _SKILLS_CACHE, _AREA_TO_SKILL
    with _LOCK:
        if _SKILLS_CACHE is not None:
            return list(_SKILLS_CACHE)  # 返回副本避免外部修改缓存

        skills = []
        _AREA_TO_SKILL.clear()

        if not os.path.isdir(SKILLS_DIR):
            print(f"警告：Skill 目录不存在 {SKILLS_DIR}")
            _SKILLS_CACHE = skills
            return list(skills)

        for fname in sorted(os.listdir(SKILLS_DIR)):
            if not fname.endswith(".md"):
                continue
            fpath = os.path.join(SKILLS_DIR, fname)
            skill = _parse_skill_md(fpath)
            if skill["name"]:
                skills.append(skill)
                _AREA_TO_SKILL[skill["area"]] = skill["name"]

        _SKILLS_CACHE = skills
        return list(skills)


def _trigger_match(question):
    """
    回退触发词匹配：在所有 skill 的 triggers 中寻找与问题最匹配的 skill。
    返回 (skill_name, hit_count)，hit_count 为命中的触发词数量。
    """
    skills = _load_skills()
    q = question.lower()

    best_name = None
    best_hits = 0

    for skill in skills:
        hits = 0
        for trigger in skill["triggers"]:
            if trigger.lower() in q:
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best_name = skill["name"]

    return best_name, best_hits


def match_skill(question):
    """
    主路径匹配：优先使用 validator.match_standard() 的 area 信号，
    若无匹配则回退用触发词匹配。

    返回:
        str or None: 匹配到的 skill 名称，未匹配则返回 None
    """
    # 主路径：利用 validator 的标准记录匹配
    match_result = validator.match_standard(question)
    if match_result["matched"] and match_result.get("best_record"):
        area = match_result["best_record"].get("area", "")
        skill_name = _AREA_TO_SKILL.get(area)
        if skill_name:
            return skill_name

    # 回退路径：触发词匹配（至少命中 1 个触发词）
    trigger_name, hits = _trigger_match(question)
    if trigger_name and hits >= 1:
        return trigger_name

    return None


def get_skill_content(skill_name):
    """
    根据 skill 名称加载其 Markdown 正文内容。

    参数:
        skill_name: skill 名称（如 "integration_management"）

    返回:
        str or None: Markdown 正文内容
    """
    if not skill_name:
        return None

    skills = _load_skills()
    for skill in skills:
        if skill["name"] == skill_name:
            return skill["body"]
    return None


def match_and_load(question):
    """
    一站式入口：匹配问题对应的 skill 并加载内容。

    参数:
        question: 用户提问

    返回:
        tuple: (skill_name, skill_content)
            - skill_name: 匹配到的 skill 名称，未匹配则为 None
            - skill_content: skill 的 Markdown 正文，未匹配则为 None
    """
    # 优先通过 skill 匹配（包含标准记录匹配 + 触发词匹配）
    # skill 匹配成功即说明属于 PMP 范畴
    skill_name = match_skill(question)
    if skill_name:
        skill_content = get_skill_content(skill_name)
        return skill_name, skill_content

    # skill 未匹配时，再用 validator.is_pmp_related 判断是否为 PMP 领域边界问题
    # 若属于 PMP 但 skill 无法精确匹配，返回 (None, None) 让 LLM 用默认提示词回答
    # 若不属于 PMP，同样返回 (None, None)，LLM 会走拒绝路径
    return None, None


def reload_skills():
    """清除缓存，强制重新加载 skill 文件。线程安全。"""
    global _SKILLS_CACHE, _AREA_TO_SKILL
    with _LOCK:
        _SKILLS_CACHE = None
        _AREA_TO_SKILL.clear()
    return _load_skills()


if __name__ == "__main__":
    # 简单自测
    print("已加载 Skills:")
    for s in _load_skills():
        print(f"  {s['name']} ({s['area']}, {s['record_count']} 条, {len(s['triggers'])} 触发词)")

    # 测试匹配
    test_questions = [
        "什么是项目章程？",
        "WBS是什么？",
        "关键路径法怎么用？",
        "挣值管理是什么？",
        "今天天气怎么样？",
    ]
    print("\n匹配测试:")
    for q in test_questions:
        name, content = match_and_load(q)
        print(f"  Q: {q} → Skill: {name} (内容长度: {len(content) if content else 0})")
