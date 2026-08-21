"""
Skill 优化对比 Benchmark

功能:
  1. 加载 3 个版本的 Skill（V1 冗长 / V2 LLM优化 / V3 人工最优）
  2. 加载 30 条测试邮件
  3. 用每个 Skill 跑全部测试，统计指标:
     - 输入 token (prompt)
     - 输出 token (completion)
     - 总 token
     - 响应时间 (秒)
     - 分类准确率
     - 输出格式合规率（能否被 json.loads 解析）
  4. 输出对比报告（控制台 + JSON + Markdown）

用法:
  python run_benchmark.py                  # 跑全部 3 个版本
  python run_benchmark.py --skill v1_verbose  # 只跑指定版本
  python run_benchmark.py --dry-run        # 不调 LLM，只显示设计（用于 API key 缺失时）
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# 强制 UTF-8 输出（Windows 默认 GBK 会乱码）
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from openai import OpenAI

ROOT = Path(__file__).parent
SKILLS_DIR = ROOT / "skills"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# 5 个合法类别
VALID_CATEGORIES = {"投诉", "建议", "咨询", "表扬", "其他"}


def load_skill(skill_path: Path) -> tuple[str, str]:
    """加载 Skill 文件，返回 (name, content)"""
    content = skill_path.read_text(encoding="utf-8")
    name = skill_path.parent.name
    return name, content


def load_test_cases() -> list[dict]:
    """加载测试邮件"""
    test_file = DATA_DIR / "test_emails.json"
    return json.loads(test_file.read_text(encoding="utf-8"))["test_cases"]


def build_system_prompt(skill_content: str) -> str:
    """构造 system prompt：包装 Skill 内容"""
    return f"""你是客户邮件分类助手。请严格根据以下 Skill 完成任务。

{skill_content}

请仅按 Skill 中要求的格式输出，不要添加任何额外内容。"""


def parse_category_from_output(raw: str) -> tuple[str, bool]:
    """
    从 LLM 原始输出中提取类别和 reason
    返回 (category, format_valid)
    """
    raw = raw.strip()

    # 尝试 1: 标准 JSON
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            # V2 格式: {"category": "...", "reason": "..."}
            # V3 格式: {"c": "...", "r": "..."}
            cat = data.get("category") or data.get("c") or ""
            if cat in VALID_CATEGORIES:
                return cat, True
    except json.JSONDecodeError:
        pass

    # 尝试 2: 提取 ```json ... ``` 块
    import re
    json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            cat = data.get("category") or data.get("c") or ""
            if cat in VALID_CATEGORIES:
                return cat, True
        except json.JSONDecodeError:
            pass

    # 尝试 3: 文本里包含类别关键词
    for cat in VALID_CATEGORIES:
        if cat in raw[:50]:  # 只看前 50 字（V1 格式："分类: 投诉"）
            return cat, False  # 不是严格 JSON

    return "", False


def run_single_skill(
    client: OpenAI,
    skill_name: str,
    skill_content: str,
    test_cases: list[dict],
    model: str = "deepseek-chat",
    dry_run: bool = False,
) -> dict:
    """用单个 Skill 跑全部测试，返回汇总指标"""
    system_prompt = build_system_prompt(skill_content)
    print(f"\n{'='*60}")
    print(f"  跑 Skill: {skill_name}")
    print(f"  Skill 长度: {len(skill_content)} 字符")
    print(f"  System prompt 长度: {len(system_prompt)} 字符")
    print(f"{'='*60}")

    if dry_run:
        # 仅估算输入 token，不实际调用
        # 1 中文字符 ≈ 1.5 token（粗略估算）
        estimated_input = int(len(system_prompt) * 1.5)
        return {
            "skill": skill_name,
            "skill_chars": len(skill_content),
            "dry_run": True,
            "estimated_prompt_tokens": estimated_input,
            "total_cases": len(test_cases),
        }

    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_time = 0.0
    correct_count = 0
    format_valid_count = 0

    for i, case in enumerate(test_cases, 1):
        user_msg = f"请分类以下邮件：\n\n{case['email']}"

        start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                max_tokens=300,
            )
            elapsed = time.perf_counter() - start
            raw_output = response.choices[0].message.content.strip()
            usage = response.usage

            in_tok = usage.prompt_tokens
            out_tok = usage.completion_tokens

            predicted, format_ok = parse_category_from_output(raw_output)
            is_correct = (predicted == case["category"])
            if is_correct:
                correct_count += 1
            if format_ok:
                format_valid_count += 1

            total_input_tokens += in_tok
            total_output_tokens += out_tok
            total_time += elapsed

            mark = "✓" if is_correct else "✗"
            fmark = "JSON✓" if format_ok else "JSON✗"
            print(f"  Q{i:02d} {mark} {fmark} | 预测={predicted or '?'} | 真值={case['category']} | {in_tok}↓{out_tok}↑ | {elapsed:.2f}s")

            results.append({
                "id": case["id"],
                "category_true": case["category"],
                "category_pred": predicted,
                "correct": is_correct,
                "format_valid": format_ok,
                "raw_output": raw_output,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "elapsed_sec": round(elapsed, 3),
            })
        except Exception as e:
            print(f"  Q{i:02d} ❌ API 错误: {e}")
            results.append({"id": case["id"], "error": str(e)})

    total = len(test_cases)
    summary = {
        "skill": skill_name,
        "skill_chars": len(skill_content),
        "system_prompt_chars": len(system_prompt),
        "total_cases": total,
        "accuracy": round(correct_count / total, 3),
        "format_valid_rate": round(format_valid_count / total, 3),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "avg_input_tokens": round(total_input_tokens / total, 1),
        "avg_output_tokens": round(total_output_tokens / total, 1),
        "total_time_sec": round(total_time, 2),
        "avg_time_sec": round(total_time / total, 3),
        "per_question": results,
    }

    print(f"\n📊 {skill_name} 汇总:")
    print(f"  准确率:    {correct_count}/{total} = {summary['accuracy']:.1%}")
    print(f"  格式合规率: {format_valid_count}/{total} = {summary['format_valid_rate']:.1%}")
    print(f"  总 token:   {summary['total_tokens']} (输入 {summary['total_input_tokens']} + 输出 {summary['total_output_tokens']})")
    print(f"  总时间:     {summary['total_time_sec']} 秒 (平均 {summary['avg_time_sec']}s/题)")
    return summary


def make_markdown_report(summaries: list[dict]) -> str:
    """生成 Markdown 格式对比报告"""
    md = ["# Skill 优化对比报告\n"]
    md.append(f"生成时间: {datetime.now().isoformat()}\n")
    md.append(f"测试样本数: {summaries[0]['total_cases']}\n\n")

    md.append("## 对比总表\n\n")
    md.append("| Skill | 字符数 | 准确率 | 格式合规率 | 总 Token | 输入 Token | 输出 Token | 平均时间 |\n")
    md.append("|-------|--------|--------|-----------|----------|-----------|-----------|---------|\n")
    for s in summaries:
        md.append(
            f"| {s['skill']} | {s['skill_chars']} | "
            f"{s['accuracy']:.1%} | {s['format_valid_rate']:.1%} | "
            f"{s.get('total_tokens', '-')} | {s.get('total_input_tokens', '-')} | "
            f"{s.get('total_output_tokens', '-')} | {s.get('avg_time_sec', '-')}s |\n"
        )

    # 优化幅度对比（V2 vs V1, V3 vs V1）
    if len(summaries) >= 2:
        v1 = summaries[0]
        md.append("\n## 优化幅度（相对 V1）\n\n")
        for s in summaries[1:]:
            char_reduction = (v1['skill_chars'] - s['skill_chars']) / v1['skill_chars']
            if s.get('total_tokens') and v1.get('total_tokens'):
                token_reduction = (v1['total_tokens'] - s['total_tokens']) / v1['total_tokens']
                token_pct = f"-{token_reduction:.1%}"
            else:
                token_pct = "N/A"
            md.append(
                f"### {s['skill']} vs V1\n"
                f"- Skill 字符数: {v1['skill_chars']} → {s['skill_chars']} ({char_reduction:+.1%})\n"
                f"- 准确率: {v1['accuracy']:.1%} → {s['accuracy']:.1%} ({(s['accuracy']-v1['accuracy']):+.1%})\n"
                f"- 格式合规率: {v1['format_valid_rate']:.1%} → {s['format_valid_rate']:.1%} ({(s['format_valid_rate']-v1['format_valid_rate']):+.1%})\n"
                f"- 总 Token: {token_pct}\n\n"
            )

    # 分类准确率细分
    md.append("## 分类准确率细分（按 5 类别）\n\n")
    for s in summaries:
        cat_stats = {c: {"total": 0, "correct": 0} for c in VALID_CATEGORIES}
        for q in s.get("per_question", []):
            if "error" in q:
                continue
            cat_stats[q["category_true"]]["total"] += 1
            if q["correct"]:
                cat_stats[q["category_true"]]["correct"] += 1
        md.append(f"### {s['skill']}\n\n")
        md.append("| 类别 | 正确/总数 | 准确率 |\n|------|----------|--------|\n")
        for cat, stat in cat_stats.items():
            if stat["total"] > 0:
                acc = stat["correct"] / stat["total"]
                md.append(f"| {cat} | {stat['correct']}/{stat['total']} | {acc:.1%} |\n")
        md.append("\n")

    return "".join(md)


def main():
    parser = argparse.ArgumentParser(description="Skill 优化对比 benchmark")
    parser.add_argument("--skill", type=str, help="只跑指定 Skill (v1_verbose / v2_llm_optimized / v3_human_optimal)")
    parser.add_argument("--dry-run", action="store_true", help="不调 API，只展示设计")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="LLM 模型名")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    # 加载测试集
    test_cases = load_test_cases()
    print(f"� 加载测试集: {len(test_cases)} 条邮件")

    # 选择 Skill
    skill_versions = ["v1_verbose", "v2_llm_optimized", "v3_human_optimal"]
    if args.skill:
        skill_versions = [args.skill]

    # 检查 API key（支持任意 OpenAI 兼容端点）
    api_key = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    # 模型名也支持环境变量覆盖
    if not args.model or args.model == "deepseek-chat":
        env_model = os.getenv("LLM_MODEL")
        if env_model:
            args.model = env_model

    if not api_key and not args.dry_run:
        print("⚠️ 未设置 LLM_API_KEY / DEEPSEEK_API_KEY，使用 --dry-run 模式")
        args.dry_run = True

    client = OpenAI(api_key=api_key or "dummy", base_url=base_url) if not args.dry_run else None

    # 跑 benchmark
    summaries = []
    for v in skill_versions:
        skill_path = SKILLS_DIR / v / "customer_classifier" / "SKILL.md"
        if not skill_path.exists():
            print(f"❌ 找不到 Skill: {skill_path}")
            continue
        name, content = load_skill(skill_path)
        summary = run_single_skill(client, name, content, test_cases,
                                    model=args.model, dry_run=args.dry_run)
        summaries.append(summary)

    # 保存结果
    if summaries:
        result_file = RESULTS_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result_file.write_text(json.dumps(summaries, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        print(f"\n💾 详细结果已保存: {result_file}")

        if not args.dry_run and len(summaries) >= 2:
            # 生成 Markdown 报告
            md = make_markdown_report(summaries)
            md_file = RESULTS_DIR / "comparison_report.md"
            md_file.write_text(md, encoding="utf-8")
            print(f"� Markdown 报告: {md_file}")


if __name__ == "__main__":
    main()
