# -*- coding: utf-8 -*-
"""
PMP 客户 FAQ 系统 - Flask 后端
功能：
1. 调用 LLM（基于 DASHSCOPE_API_KEY）作为 PMP 专家回答用户提问；
2. 每次问答按日期保存历史提问、回答信息（history/YYYY-MM-DD.json）；
3. 依据 PMP 问答标准体系判定提问与回答的正确性；
4. 未满足问答标准体系的统一保存到错误文件，按日期分类（errors/YYYY-MM-DD.json）；
5. 提供历史/错误记录查询接口。
"""
import os
import json
from datetime import datetime
from collections import Counter

from flask import Flask, request, jsonify, render_template

import llm_client
import validator
import skill_loader
import skill_optimizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_DIR = os.path.join(BASE_DIR, "history")
ERRORS_DIR = os.path.join(BASE_DIR, "errors")

app = Flask(__name__, template_folder="templates", static_folder="static")

# 启动时确保目录存在
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(ERRORS_DIR, exist_ok=True)


def _today_str():
    """返回当前日期字符串 YYYY-MM-DD"""
    return datetime.now().strftime("%Y-%m-%d")


def _now_str():
    """返回当前时间字符串 YYYY-MM-DD HH:MM:SS"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _append_jsonl(filepath, record):
    """以 JSON Lines 方式追加一条记录到文件（每行一个 JSON 对象）。"""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(filepath):
    """读取 JSON Lines 文件，返回记录列表。"""
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


def _list_dates(directory):
    """列出目录中所有日期文件（去扩展名），按日期降序。"""
    dates = []
    if not os.path.isdir(directory):
        return dates
    for fname in os.listdir(directory):
        if fname.endswith(".jsonl"):
            dates.append(fname[:-6])  # 去掉 .jsonl
    dates.sort(reverse=True)
    return dates


def _determine_phase():
    """
    判断当前阶段：pre-optimization / post-optimization。
    基于 optimization_log.json 中最近一次优化时间判断。
    """
    try:
        logs = skill_optimizer.get_optimization_log()
        if logs:
            last_log = logs[-1]
            optimize_time = datetime.strptime(last_log["timestamp"], "%Y-%m-%d %H:%M:%S")
            now = datetime.now()
            if now > optimize_time:
                return "post-optimization"
    except Exception:
        pass
    return "pre-optimization"


@app.route("/")
def index():
    """主页：实时问答交互页面"""
    return render_template("index.html")


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """
    问答接口。
    请求体: {"question": "用户问题"}
    流程:
      1) 调用 LLM 获取回答；
      2) 用验证器评估问答是否符合 PMP 标准体系；
      3) 符合标准 => 追加保存到 history/YYYY-MM-DD.jsonl；
         不符合标准 => 追加保存到 errors/YYYY-MM-DD.jsonl；
         （历史记录中也会保留所有问答，错误文件单独记录不符合标准的）
    """
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"success": False, "error": "问题不能为空"}), 400

    # 1. 技能匹配：根据提问确定对应的 skill 并加载上下文
    skill_name, skill_content = skill_loader.match_and_load(question)

    # 2. 调用 LLM（注入匹配到的 skill 上下文）
    llm_result = llm_client.chat(question, skill_content=skill_content)
    if not llm_result["success"]:
        return jsonify({
            "success": False,
            "error": f"LLM 调用失败：{llm_result['error']}",
        }), 502

    answer = llm_result["answer"]
    model = llm_result["model"]
    prompt_tokens = llm_result.get("prompt_tokens", 0)
    completion_tokens = llm_result.get("completion_tokens", 0)
    total_tokens = llm_result.get("total_tokens", 0)

    # 3. 验证问答是否符合 PMP 标准体系
    evaluation = validator.evaluate(question, answer)

    # 判断当前阶段（优化前/优化后）
    phase = _determine_phase()

    # 4. 构造一条完整的问答记录
    record = {
        "timestamp": _now_str(),
        "date": _today_str(),
        "question": question,
        "answer": answer,
        "model": model,
        "skill": skill_name,
        "evaluation": evaluation,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "phase": phase,
    }

    # 5. 按日期保存
    #    - 历史文件：保存所有问答（符合与不符合都保留，便于追溯）
    #    - 错误文件：仅保存未满足问答标准体系的记录（按日期分类）
    history_file = os.path.join(HISTORY_DIR, f"{_today_str()}.jsonl")
    _append_jsonl(history_file, record)

    if not evaluation["valid"]:
        error_file = os.path.join(ERRORS_DIR, f"{_today_str()}.jsonl")
        _append_jsonl(error_file, record)

    # 6. 返回给前端
    return jsonify({
        "success": True,
        "question": question,
        "answer": answer,
        "model": model,
        "skill": skill_name,
        "evaluation": evaluation,
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
        },
        "phase": phase,
        "saved_to": "history" if evaluation["valid"] else "history + errors",
    })


@app.route("/api/history")
@app.route("/api/history/<date>")
def api_history(date=None):
    """查询历史问答记录。不指定 date 时返回日期列表。"""
    if date is None:
        return jsonify({"success": True, "dates": _list_dates(HISTORY_DIR)})
    filepath = os.path.join(HISTORY_DIR, f"{date}.jsonl")
    return jsonify({"success": True, "date": date, "records": _read_jsonl(filepath)})


@app.route("/api/errors")
@app.route("/api/errors/<date>")
def api_errors(date=None):
    """查询未满足标准体系的错误记录。不指定 date 时返回日期列表。"""
    if date is None:
        return jsonify({"success": True, "dates": _list_dates(ERRORS_DIR)})
    filepath = os.path.join(ERRORS_DIR, f"{date}.jsonl")
    return jsonify({"success": True, "date": date, "records": _read_jsonl(filepath)})


@app.route("/api/stats")
def api_stats():
    """统计：PMP 标准体系概览 + 历史问答统计。"""
    records = validator.get_standard_records()
    area_counter = Counter(r["area"] for r in records)

    # 历史统计
    history_total = 0
    for d in _list_dates(HISTORY_DIR):
        history_total += len(_read_jsonl(os.path.join(HISTORY_DIR, f"{d}.jsonl")))

    # 错误统计
    error_total = 0
    for d in _list_dates(ERRORS_DIR):
        error_total += len(_read_jsonl(os.path.join(ERRORS_DIR, f"{d}.jsonl")))

    return jsonify({
        "success": True,
        "standard": {
            "total": len(records),
            "areas": dict(area_counter),
        },
        "history_total": history_total,
        "error_total": error_total,
        "history_dates": _list_dates(HISTORY_DIR),
        "error_dates": _list_dates(ERRORS_DIR),
    })


@app.route("/api/standard")
def api_standard():
    """查看 PMP 问答标准体系（按知识领域分组）。"""
    records = validator.get_standard_records()
    grouped = {}
    for r in records:
        grouped.setdefault(r["area"], []).append({
            "id": r["id"],
            "question": r["question"],
            "standard_answer": r["standard_answer"],
            "keywords": r.get("keywords", []),
            "process_group": r.get("process_group", ""),
        })
    return jsonify({
        "success": True,
        "total": len(records),
        "areas": grouped,
    })


@app.route("/api/optimize-status")
def api_optimize_status():
    """获取 Skill 优化状态：错误统计、Skill 信息、优化日志。"""
    # 收集错误统计
    skill_errors = skill_optimizer._collect_errors()
    error_stats = []
    total_errors = 0
    for skill_name, records in skill_errors.items():
        skill_info = skill_loader.get_skill_content(skill_name)
        error_stats.append({
            "skill": skill_name,
            "error_count": len(records),
            "area": next((s["area"] for s in skill_loader._load_skills() if s["name"] == skill_name), ""),
        })
        total_errors += len(records)

    # Skill 信息
    skills_info = []
    for s in skill_loader._load_skills():
        # 检查是否有备份
        backup_dir = skill_optimizer.BACKUP_DIR
        backup_count = 0
        if os.path.isdir(backup_dir):
            backup_count = len([f for f in os.listdir(backup_dir) if f.startswith(s["name"])])

        skills_info.append({
            "name": s["name"],
            "area": s["area"],
            "record_count": s["record_count"],
            "trigger_count": len(s["triggers"]),
            "body_length": len(s["body"]),
            "backup_count": backup_count,
        })

    # 优化日志
    logs = skill_optimizer.get_optimization_log()

    return jsonify({
        "success": True,
        "total_errors": total_errors,
        "error_by_skill": error_stats,
        "skills": skills_info,
        "recent_logs": logs[-5:] if logs else [],
    })


@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    """
    执行 Skill 优化。
    请求体: {"skill_name": "integration_management"}  可选，指定优化的 skill
    """
    data = request.get_json(silent=True) or {}
    target_skill = (data.get("skill_name") or "").strip() or None

    try:
        result = skill_optimizer.run_optimization(target_skill=target_skill)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"优化过程出错: {str(e)}",
            "results": [],
        }), 500


@app.route("/api/token-stats")
def api_token_stats():
    """
    获取 Token 消耗统计数据，用于趋势图展示。
    
    返回数据结构：
    {
        "success": True,
        "summary": {
            "total_tokens": 12345,
            "total_prompt_tokens": 5000,
            "total_completion_tokens": 7345,
            "total_queries": 100,
            "avg_tokens_per_query": 123,
            "pre_optimization_total": 5000,
            "post_optimization_total": 7345,
        },
        "trend": [
            {"date": "2026-08-06", "time": "10:00:00", "total_tokens": 150, "phase": "pre-optimization"},
            {"date": "2026-08-06", "time": "10:05:00", "total_tokens": 200, "phase": "post-optimization"},
            ...
        ],
        "by_skill": {
            "integration_management": 1234,
            ...
        },
        "by_date": {
            "2026-08-06": 3000,
            "2026-08-05": 5000,
        }
    }
    """
    all_records = []
    for date in _list_dates(HISTORY_DIR):
        filepath = os.path.join(HISTORY_DIR, f"{date}.jsonl")
        records = _read_jsonl(filepath)
        all_records.extend(records)

    # 汇总统计
    total_tokens = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    pre_opt_total = 0
    post_opt_total = 0
    by_skill = {}
    by_date = {}

    # 趋势数据
    trend = []

    for rec in all_records:
        tokens = rec.get("total_tokens", 0) or 0
        prompt_t = rec.get("prompt_tokens", 0) or 0
        completion_t = rec.get("completion_tokens", 0) or 0
        skill = rec.get("skill") or "unknown"
        date = rec.get("date", "") or "unknown"
        timestamp = rec.get("timestamp", "") or ""
        phase = rec.get("phase") or "pre-optimization"

        total_tokens += tokens
        total_prompt_tokens += prompt_t
        total_completion_tokens += completion_t

        if phase == "pre-optimization":
            pre_opt_total += tokens
        else:
            post_opt_total += tokens

        # 按 skill 统计
        if skill not in by_skill:
            by_skill[skill] = 0
        by_skill[skill] += tokens

        # 按日期统计
        if date not in by_date:
            by_date[date] = 0
        by_date[date] += tokens

        # 趋势点
        trend_point = {
            "date": date,
            "time": timestamp.split(" ")[-1] if timestamp else "",
            "total_tokens": tokens,
            "phase": phase,
            "skill": skill,
            "question_preview": rec.get("question", "")[:30],
        }
        trend.append(trend_point)

    # 按时间排序
    trend.sort(key=lambda x: f"{x['date']} {x['time']}")

    total_queries = len(all_records)
    avg_tokens = round(total_tokens / total_queries) if total_queries > 0 else 0

    return jsonify({
        "success": True,
        "summary": {
            "total_tokens": total_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_queries": total_queries,
            "avg_tokens_per_query": avg_tokens,
            "pre_optimization_total": pre_opt_total,
            "post_optimization_total": post_opt_total,
            "optimization_improvement": round((1 - post_opt_total / max(pre_opt_total, 1)) * 100, 1) if pre_opt_total > 0 else 0,
        },
        "trend": trend[-100:],  # 最近100条
        "by_skill": by_skill,
        "by_date": dict(sorted(by_date.items(), reverse=True)),
    })


@app.route("/health")
def health():
    """健康检查。"""
    try:
        has_key = bool(os.environ.get("DASHSCOPE_API_KEY"))
    except Exception:
        has_key = False
    return jsonify({
        "success": True,
        "status": "ok",
        "dashscope_api_key_configured": has_key,
        "standard_records": len(validator.get_standard_records()),
    })


if __name__ == "__main__":
    # 启动前打印一些信息
    print("=" * 60)
    print("PMP 客户 FAQ 系统")
    print(f"  标准记录数: {len(validator.get_standard_records())}")
    print(f"  DASHSCOPE_API_KEY: {'已配置' if os.environ.get('DASHSCOPE_API_KEY') else '未配置!'}")
    print(f"  历史目录: {HISTORY_DIR}")
    print(f"  错误目录: {ERRORS_DIR}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False, threaded=True)
