# -*- coding: utf-8 -*-
"""
Skill 优化器：基于 DASHSCOPE_API_KEY 调用 LLM，从错误记录中学习，
自动优化 pmp_skills/*.md 文件。

核心逻辑：
1. 读取 errors/ 目录中的错误问答记录
2. 调用 LLM 分析错误回答，生成结构化优化建议（JSON 格式）
3. 将优化建议合并到对应的 skill.md 文件中（新增 triggers、补充 Q&A 对）
4. 自动备份原文件，防止数据损坏
5. 刷新 skill_loader 缓存，使优化立即生效
"""
import os
import json
import shutil
import re
from datetime import datetime

import llm_client
import skill_loader
import validator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ERRORS_DIR = os.path.join(BASE_DIR, "errors")
SKILLS_DIR = os.path.join(BASE_DIR, "pmp_skills")
BACKUP_DIR = os.path.join(BASE_DIR, "pmp_skills_backup")
OPTIMIZATION_LOG = os.path.join(BASE_DIR, "optimization_log.json")


def _read_jsonl(filepath):
    """读取 JSON Lines 文件。"""
    records = []
    if not os.path.exists(filepath):
        return records
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _collect_errors():
    """
    收集所有错误记录，按 skill 分组。
    对于 skill 为 None 的记录，通过 skill_loader.match_skill 补充匹配。

    返回:
        dict: {skill_name: [record, ...]}
    """
    skill_errors = {}

    if not os.path.isdir(ERRORS_DIR):
        return skill_errors

    for fname in sorted(os.listdir(ERRORS_DIR)):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(ERRORS_DIR, fname)
        for record in _read_jsonl(fpath):
            # 只处理属于 PMP 领域但未满足标准体系的记录
            ev = record.get("evaluation", {})
            if ev.get("is_pmp") and not ev.get("valid"):
                skill_name = record.get("skill")
                # 如果 skill 为空，尝试重新匹配
                if not skill_name:
                    skill_name = skill_loader.match_skill(record.get("question", ""))
                if skill_name:
                    if skill_name not in skill_errors:
                        skill_errors[skill_name] = []
                    skill_errors[skill_name].append(record)

    return skill_errors


def _build_optimization_prompt(skill_name, skill_body, error_records):
    """
    构建 Skill 优化的系统提示词。

    参数:
        skill_name: skill 名称
        skill_body: 当前 skill 的 Markdown 正文（不含 front matter）
        error_records: 该 skill 相关的错误记录列表

    返回:
        str: 完整的用户提示词
    """
    error_summary = []
    for rec in error_records:
        ev = rec.get("evaluation", {})
        error_summary.append(
            f"- **问题**: {rec.get('question', '')}\n"
            f"  **LLM 回答**: {rec.get('answer', '')[:200]}\n"
            f"  **错误原因**: {ev.get('reason', '未知')}"
        )

    error_text = "\n".join(error_summary)

    prompt = f"""你是一位 Skill 优化专家。请根据以下信息，优化 PMP 知识领域 Skill 的内容。

## 当前 Skill: {skill_name}

### 当前 Skill 内容摘要：
{skill_body[:3000] if skill_body else '（暂无内容）'}

### 错误问答记录（共 {len(error_records)} 条）：
{error_text}

## 任务要求：
请分析以上错误记录，识别 LLM 回答中缺失或不准确的知识点，并输出结构化的优化建议。

你必须严格按照以下 JSON 格式返回结果（不要返回 Markdown 或其他格式）：

```json
{{
  "analysis": "对错误原因的简要分析",
  "new_triggers": ["新增的触发词1", "新增的触发词2"],
  "new_qa": [
    {{
      "question": "需要补充的标准问题",
      "standard_answer": "该问题的标准答案（100字以内，简明扼要）",
      "keywords": ["关键词1", "关键词2"]
    }}
  ],
  "suggestions": ["对 Skill 内容优化的文字建议"]
}}
```

注意：
1. `new_triggers` 为 0-10 个字符串列表，用于增强触发词匹配
2. `new_qa` 为 0-5 个新的标准问答对，用于补充 Skill 知识库
3. 如果某个字段不需要新增，请返回空数组
4. 回答要基于 PMBOK 指南，确保专业性
5. 不要返回 JSON 以外的任何文字"""

    return prompt


def _parse_llm_response(response_text):
    """
    从 LLM 响应中提取 JSON 优化建议。

    参数:
        response_text: LLM 返回的文本

    返回:
        dict: {"analysis": str, "new_triggers": list, "new_qa": list, "suggestions": list}
    """
    default = {
        "analysis": "",
        "new_triggers": [],
        "new_qa": [],
        "suggestions": [],
    }

    if not response_text:
        return default

    # 尝试从文本中提取 JSON 块
    json_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # 尝试直接解析整个文本
        json_str = response_text.strip()
        if not json_str.startswith("{"):
            # 尝试找到第一个 { 开始的位置
            idx = json_str.find("{")
            if idx == -1:
                return default
            json_str = json_str[idx:]

    try:
        data = json.loads(json_str)
        return {
            "analysis": data.get("analysis", ""),
            "new_triggers": data.get("new_triggers", []),
            "new_qa": data.get("new_qa", []),
            "suggestions": data.get("suggestions", []),
        }
    except (json.JSONDecodeError, ValueError):
        return default


def _backup_skill_file(skill_name):
    """
    备份指定的 skill 文件到 pmp_skills_backup/ 目录。

    参数:
        skill_name: skill 名称
    """
    os.makedirs(BACKUP_DIR, exist_ok=True)
    skill_file = os.path.join(SKILLS_DIR, f"{skill_name}.md")
    if os.path.exists(skill_file):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(BACKUP_DIR, f"{skill_name}_{timestamp}.md")
        shutil.copy2(skill_file, backup_file)


def _update_skill_file(skill_name, optimization):
    """
    将优化建议合并到 skill.md 文件中。

    参数:
        skill_name: skill 名称
        optimization: LLM 返回的优化建议 dict

    返回:
        dict: 更新结果 {"triggers_added": int, "qa_added": int}
    """
    skill_file = os.path.join(SKILLS_DIR, f"{skill_name}.md")
    if not os.path.exists(skill_file):
        return {"triggers_added": 0, "qa_added": 0}

    with open(skill_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 解析现有内容
    meta = skill_loader._parse_yaml_front_matter(content)
    body = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, count=1, flags=re.DOTALL)

    existing_triggers = set(meta.get("triggers", []))
    existing_qa_ids = set()
    for line in body.split("\n"):
        m = re.match(r"###\s+(\w+)-(\d+)", line)
        if m:
            existing_qa_ids.add(f"{m.group(1)}-{m.group(2)}")

    # 1. 新增 triggers
    new_triggers = []
    for t in optimization.get("new_triggers", []):
        t = str(t).strip()
        if t and t not in existing_triggers:
            new_triggers.append(t)
            existing_triggers.add(t)

    # 2. 新增 Q&A 对
    new_qa_entries = []
    base_id_prefix = skill_name[:3].upper() if skill_name else "OPT"
    existing_nums = []
    for line in body.split("\n"):
        m = re.match(r"###\s+\w+-(\d+)", line)
        if m:
            existing_nums.append(int(m.group(1)))

    next_num = max(existing_nums) + 1 if existing_nums else 31  # 从 31 开始编号（前30条为标准条目）

    for qa in optimization.get("new_qa", []):
        q = qa.get("question", "").strip()
        a = qa.get("standard_answer", "").strip()
        kws = qa.get("keywords", [])
        if q and a:
            qa_id = f"{base_id_prefix}-{next_num:03d}"
            qa_block = f"### {qa_id}. {q}\n\n"
            qa_block += f"**标准答案**：{a}\n\n"
            if kws:
                qa_block += f"**关键词**：{'、'.join(kws)}\n\n"
            qa_block += f"**过程组**：优化补充\n\n"
            new_qa_entries.append(qa_block)
            next_num += 1

    # 3. 更新 front matter
    meta["triggers"] = sorted(existing_triggers)
    meta["record_count"] = meta.get("record_count", 0) + len(new_qa_entries)

    # 构建新的 front matter
    lines = ["---"]
    lines.append(f"name: {meta.get('name', skill_name)}")
    lines.append(f"area: {meta.get('area', '')}")
    lines.append(f"area_en: {meta.get('area_en', '')}")
    lines.append(f"record_count: {meta['record_count']}")
    lines.append("triggers:")
    for t in meta["triggers"]:
        lines.append(f'  - "{t}"')
    lines.append("process_groups:")
    for pg in meta.get("process_groups", []):
        lines.append(f'  - "{pg}"')
    lines.append("---")
    new_front_matter = "\n".join(lines)

    # 4. 合并内容
    new_body = body.strip()
    if new_qa_entries:
        new_body += "\n\n## 优化补充问答\n\n" + "\n".join(new_qa_entries)
    if new_triggers:
        # 不直接在 body 中重复 triggers，已在 front matter 中更新
        pass

    new_content = new_front_matter + "\n" + new_body + "\n"

    # 5. 备份后写入
    _backup_skill_file(skill_name)
    with open(skill_file, "w", encoding="utf-8") as f:
        f.write(new_content)

    return {
        "triggers_added": len(new_triggers),
        "qa_added": len(new_qa_entries),
        "total_triggers": len(meta["triggers"]),
        "total_records": meta["record_count"],
    }


def optimize_skill(skill_name, error_records):
    """
    对单个 skill 执行优化流程。

    参数:
        skill_name: skill 名称
        error_records: 该 skill 相关的错误记录列表

    返回:
        dict: 优化结果
    """
    # 获取当前 skill 内容
    skill_body = skill_loader.get_skill_content(skill_name) or ""

    # 构建优化提示词
    prompt = _build_optimization_prompt(skill_name, skill_body, error_records)

    # 调用 LLM
    result = llm_client.chat(prompt, temperature=0.3)
    if not result["success"]:
        return {
            "skill": skill_name,
            "success": False,
            "error": f"LLM 调用失败：{result['error']}",
        }

    # 解析 LLM 返回的优化建议
    optimization = _parse_llm_response(result["answer"])

    # 合并到 skill 文件
    update_result = _update_skill_file(skill_name, optimization)

    return {
        "skill": skill_name,
        "success": True,
        "analysis": optimization.get("analysis", ""),
        "suggestions": optimization.get("suggestions", []),
        "updates": update_result,
        "error_count": len(error_records),
    }


def run_optimization(target_skill=None):
    """
    执行 Skill 优化流程。

    参数:
        target_skill: 指定优化的 skill 名称（None 则优化所有有错误的 skill）

    返回:
        dict: 优化结果汇总
    """
    # 收集错误记录
    skill_errors = _collect_errors()

    if not skill_errors:
        return {
            "success": True,
            "message": "暂无 PMP 领域的错误记录，无需优化。",
            "results": [],
        }

    # 确定要优化的 skill 列表
    if target_skill:
        if target_skill not in skill_errors:
            return {
                "success": False,
                "message": f"Skill '{target_skill}' 没有相关错误记录。",
                "results": [],
            }
        skills_to_optimize = {target_skill: skill_errors[target_skill]}
    else:
        skills_to_optimize = skill_errors

    # 执行优化
    results = []
    for skill_name, records in skills_to_optimize.items():
        print(f"正在优化 Skill: {skill_name}（{len(records)} 条错误记录）...")
        result = optimize_skill(skill_name, records)
        results.append(result)

    # 刷新 skill 缓存
    skill_loader.reload_skills()

    # 记录优化日志
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target": target_skill or "all",
        "results": results,
    }
    _append_optimization_log(log_entry)

    # 汇总
    total_triggers = sum(r.get("updates", {}).get("triggers_added", 0) for r in results)
    total_qa = sum(r.get("updates", {}).get("qa_added", 0) for r in results)
    total_errors = sum(r.get("error_count", 0) for r in results)

    return {
        "success": True,
        "message": f"优化完成！共处理 {len(results)} 个 Skill，新增 {total_triggers} 个触发词，{total_qa} 条问答对。",
        "skills_optimized": len(results),
        "total_triggers_added": total_triggers,
        "total_qa_added": total_qa,
        "total_errors_analyzed": total_errors,
        "results": results,
    }


def _append_optimization_log(log_entry):
    """追加优化日志到文件。"""
    logs = []
    if os.path.exists(OPTIMIZATION_LOG):
        with open(OPTIMIZATION_LOG, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []

    logs.append(log_entry)
    # 只保留最近 50 条
    logs = logs[-50:]

    with open(OPTIMIZATION_LOG, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def get_optimization_log():
    """获取优化日志。"""
    if not os.path.exists(OPTIMIZATION_LOG):
        return []
    with open(OPTIMIZATION_LOG, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


if __name__ == "__main__":
    # 测试：查看当前错误统计
    skill_errors = _collect_errors()
    if skill_errors:
        print(f"发现 {sum(len(v) for v in skill_errors.values())} 条错误记录，分布在 {len(skill_errors)} 个 Skill：")
        for name, records in skill_errors.items():
            print(f"  {name}: {len(records)} 条")
    else:
        print("暂无 PMP 领域的错误记录。")
