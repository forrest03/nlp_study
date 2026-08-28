#!/usr/bin/env python3
"""试跑 weekly-report-generator skill（v1/v2 通用）。
v2 动态Few-shot：input<80字符时自动追加1个示例（模拟按需加载），如实计入prompt tokens。
带计时与 --quiet（仅打印摘要，不回显正文）。"""
import os
import sys
import time
import argparse
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = {
    "v1": "skills/weekly-report-generator/SKILL.md",
    "v2": "skills/weekly-report-generator-v2/SKILL.md",
}
FEW_SHOT = "skills/weekly-report-generator-v2/references/few_shot.md"


def load(path):
    with open(os.path.join(HERE, path), encoding="utf-8") as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v1", "v2"], required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--quiet", action="store_true", help="不打印正文，仅打印摘要（计时/Token）")
    args = ap.parse_args()

    user_input = args.input
    system_prompt = load(SKILL[args.version])
    few_shot_used = False

    # v2 动态Few-shot：input<80字符 -> 追加1个示例（取示例1，模拟"加载1个最贴近示例"）
    if args.version == "v2" and len(user_input) < 80:
        ref = load(FEW_SHOT)
        example1 = ref.split("---")[0].strip()
        system_prompt = system_prompt + "\n\n[动态Few-shot：输入<80字符，已加载1个示例参考]\n" + example1
        few_shot_used = True

    if os.environ.get("DEEPSEEK_API_KEY"):
        client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        model, provider = "deepseek-chat", "deepseek/deepseek-chat"
    elif os.environ.get("DASHSCOPE_API_KEY"):
        client = OpenAI(api_key=os.environ["DASHSCOPE_API_KEY"],
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        model, provider = "qwen-plus", "dashscope/qwen-plus"
    else:
        sys.exit("ERROR: 未找到 DEEPSEEK_API_KEY / DASHSCOPE_API_KEY")

    print(f"[tag] {args.tag or '-'} | [version] {args.version} | few_shot={few_shot_used} | input_chars={len(user_input)} | sys_chars={len(system_prompt)}")

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0.7,
        stream=False,
    )
    elapsed = time.perf_counter() - t0

    content = resp.choices[0].message.content
    if not args.quiet:
        print(content)

    tag_slug = (args.tag or args.version).replace(" ", "_")
    out_path = os.path.join(HERE, f"sample_report_{args.version}_{tag_slug}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    u = resp.usage
    print(f"Prompt Tokens: {u.prompt_tokens}")
    print(f"Completion Tokens: {u.completion_tokens}")
    print(f"Total Tokens: {u.total_tokens}")
    print(f"Elapsed (s): {elapsed:.2f}")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
