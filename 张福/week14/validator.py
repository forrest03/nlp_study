# -*- coding: utf-8 -*-
"""
PMP 问答验证器：
1. 判定用户提问是否属于 PMP 知识领域；
2. 将提问与 PMP 标准体系进行匹配（关键词 + 文本相似度）；
3. 判定 LLM 回答是否符合问答标准体系；
4. 没有满足问答标准体系的，标记为错误（由 app.py 保存到错误文件）。
"""
import os
import json
import re
from difflib import SequenceMatcher

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STANDARD_DIR = os.path.join(BASE_DIR, "pmp_standard")

# 拒绝话术（非 PMP 问题时 LLM 的标准回复）
REJECT_REPLY = "很抱歉，当前并不能回答您PMP以外的知识领域知识。"

# PMP 知识领域名称（中英文），用于初步判断是否属于 PMP
PMP_AREA_NAMES = [
    "整体管理", "范围管理", "进度管理", "成本管理", "质量管理",
    "资源管理", "沟通管理", "采购管理", "风险管理", "相关方管理",
    "项目章程", "WBS", "挣值", "EVM", "关键路径", "CCB", "变更控制",
    "PMBOK", "PMP", "项目管理", "项目经理", "过程组", "启动", "规划",
    "执行", "监控", "收尾", "可交付成果", "相关方", "干系人",
    "里程碑", "甘特图", "风险", "采购", "合同", "质量", "沟通",
    "团队", "资源", "成本", "预算", "进度", "范围",
]


def _load_standard():
    """加载所有 PMP 标准问答记录，返回列表。"""
    records = []
    if not os.path.isdir(STANDARD_DIR):
        return records
    for fname in sorted(os.listdir(STANDARD_DIR)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(STANDARD_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        area = data.get("area", "")
        for rec in data.get("records", []):
            rec = dict(rec)
            rec["area"] = area
            records.append(rec)
    return records


# 模块级缓存，避免每次请求都读取磁盘
_STANDARD_CACHE = None


def get_standard_records():
    global _STANDARD_CACHE
    if _STANDARD_CACHE is None:
        _STANDARD_CACHE = _load_standard()
    return _STANDARD_CACHE


def _normalize(text):
    """简单归一化：去标点、空格，便于相似度比较。"""
    if not text:
        return ""
    text = str(text)
    text = re.sub(r"[\s，。、；：！？,.;:!?'\"（）()\[\]【】{}]", "", text)
    return text.lower()


def _similarity(a, b):
    """基于 SequenceMatcher 的相似度（0~1）。"""
    a = _normalize(a)
    b = _normalize(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _keyword_overlap(question, record):
    """计算问题与标准记录关键词的重叠数。"""
    q = _normalize(question)
    hits = 0
    for kw in record.get("keywords", []):
        if _normalize(kw) in q:
            hits += 1
    return hits


def is_pmp_related(question):
    """
    初步判定问题是否属于 PMP 知识领域。
    规则：问题中是否出现 PMP 领域关键词。
    """
    q = _normalize(question)
    if not q:
        return False
    for name in PMP_AREA_NAMES:
        if _normalize(name) in q:
            return True
    return False


def match_standard(question, threshold=0.25):
    """
    在 PMP 标准体系中匹配与用户提问最相似的记录。

    返回:
        dict: {
            "matched": bool,           # 是否匹配到标准记录
            "best_record": dict/None,  # 最匹配的标准记录
            "similarity": float,       # 相似度
            "keyword_hits": int,       # 关键词命中数
            "score": float,            # 综合得分
        }
    """
    records = get_standard_records()
    best = None
    best_score = 0.0
    best_sim = 0.0
    best_kw = 0

    for rec in records:
        sim = _similarity(question, rec.get("question", ""))
        kw = _keyword_overlap(question, rec)
        # 综合得分：相似度为主，关键词命中加权
        score = sim + 0.08 * kw
        if score > best_score:
            best_score = score
            best_sim = sim
            best_kw = kw
            best = rec

    matched = best is not None and (best_sim >= threshold or best_kw >= 2)
    return {
        "matched": matched,
        "best_record": best,
        "similarity": round(best_sim, 3),
        "keyword_hits": best_kw,
        "score": round(best_score, 3),
    }


def validate_answer(answer, matched_record):
    """
    判定 LLM 回答是否符合问答标准体系。

    判定逻辑：
    1. 若回答是拒绝话术（说明问题非 PMP），返回 invalid（非 PMP 问答）。
    2. 若匹配到标准记录，综合"文本相似度"与"标准答案关键词覆盖率"判定：
       - 标准答案关键词覆盖率 = 回答中包含的标准答案关键词比例；
       - 当 相似度 >= 0.15 或 关键词覆盖率 >= 0.4 时，视为符合标准体系；
       这样能正确识别 LLM 自由发挥但要点正确的回答。
    3. 若未匹配到标准记录但回答不是拒绝话术，视为未满足标准体系。

    返回:
        dict: {
            "valid": bool,             # 是否符合 PMP 问答标准体系
            "reason": str,             # 判定原因
            "answer_similarity": float,# 回答与标准答案相似度
            "keyword_coverage": float, # 标准答案关键词覆盖率
        }
    """
    # 回答为拒绝话术 => 问题非 PMP，不纳入标准体系
    if answer and answer.strip().startswith(REJECT_REPLY[:12]):
        return {
            "valid": False,
            "reason": "非PMP知识领域问题（系统已拒绝回答）",
            "answer_similarity": 0.0,
            "keyword_coverage": 0.0,
        }

    if matched_record is None:
        return {
            "valid": False,
            "reason": "提问未匹配到PMP问答标准体系中的任何记录",
            "answer_similarity": 0.0,
            "keyword_coverage": 0.0,
        }

    sim = _similarity(answer, matched_record.get("standard_answer", ""))
    coverage = _keyword_coverage(answer, matched_record)

    area = matched_record.get("area", "")
    # 综合判定：相似度达标 或 关键词覆盖达标，即视为符合标准体系
    if sim >= 0.15 or coverage >= 0.4:
        reason = (f"符合PMP标准体系（知识领域: {area}，"
                  f"相似度 {sim:.2f}，关键词覆盖率 {coverage:.0%}）")
        return {
            "valid": True,
            "reason": reason,
            "answer_similarity": round(sim, 3),
            "keyword_coverage": round(coverage, 3),
        }
    else:
        reason = (f"回答与PMP标准答案偏差较大（相似度 {sim:.2f}，"
                  f"关键词覆盖率 {coverage:.0%}，均低于阈值）")
        return {
            "valid": False,
            "reason": reason,
            "answer_similarity": round(sim, 3),
            "keyword_coverage": round(coverage, 3),
        }


def _keyword_coverage(answer, record):
    """
    计算标准记录关键词在回答中的覆盖率（0~1）。
    覆盖率 = 回答中包含的关键词数 / 关键词总数。
    """
    keywords = record.get("keywords", [])
    if not keywords:
        return 0.0
    ans = _normalize(answer)
    if not ans:
        return 0.0
    hits = sum(1 for kw in keywords if _normalize(kw) in ans)
    return hits / len(keywords)


def evaluate(question, answer):
    """
    对一次问答进行完整评估（匹配标准 + 验证回答）。

    返回:
        dict: {
            "is_pmp": bool,                 # 是否属于 PMP 范畴
            "matched": bool,                # 是否匹配到标准记录
            "valid": bool,                  # 是否符合标准体系
            "area": str/None,               # 所属知识领域
            "standard_id": str/None,        # 匹配的标准记录 ID
            "standard_question": str/None,  # 匹配的标准问题
            "standard_answer": str/None,    # 标准答案（供前端参考）
            "similarity": float,            # 提问相似度
            "answer_similarity": float,     # 回答相似度
            "reason": str,                  # 判定原因
        }
    """
    is_pmp = is_pmp_related(question)
    match_result = match_standard(question)
    matched_record = match_result["best_record"] if match_result["matched"] else None

    valid_result = validate_answer(answer, matched_record)

    return {
        "is_pmp": is_pmp,
        "matched": match_result["matched"],
        "valid": valid_result["valid"],
        "area": matched_record.get("area") if matched_record else None,
        "standard_id": matched_record.get("id") if matched_record else None,
        "standard_question": matched_record.get("question") if matched_record else None,
        "standard_answer": matched_record.get("standard_answer") if matched_record else None,
        "similarity": match_result["similarity"],
        "answer_similarity": valid_result["answer_similarity"],
        "keyword_coverage": valid_result.get("keyword_coverage", 0.0),
        "reason": valid_result["reason"],
    }


if __name__ == "__main__":
    # 自测
    records = get_standard_records()
    print(f"标准记录数: {len(records)}")
    print("领域分布:")
    from collections import Counter
    print(Counter(r["area"] for r in records))
    print("---")
    print("评估测试:")
    print(json.dumps(evaluate("什么是项目章程？", "项目章程是发起人签发的文件。"), ensure_ascii=False, indent=2))
    print(json.dumps(evaluate("今天天气怎么样？", REJECT_REPLY), ensure_ascii=False, indent=2))
