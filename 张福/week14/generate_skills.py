# -*- coding: utf-8 -*-
"""
从 pmp_standard/*.json 生成 pmp_skills/*.md（10 个 Skill 文件）。

每个 skill.md 格式：
  - YAML front matter（name / area / area_en / triggers / record_count / process_groups）
  - Markdown 正文（领域概述 + 30 条标准问答）
"""
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STANDARD_DIR = os.path.join(BASE_DIR, "pmp_standard")
SKILLS_DIR = os.path.join(BASE_DIR, "pmp_skills")

# 每个知识领域额外补充的触发词（领域级关键词，不依赖 JSON 中的 keywords）
AREA_EXTRA_TRIGGERS = {
    "项目整体管理": ["整体", "整合", "章程", "变更控制", "CCB", "收尾", "项目管理计划", "基准", "整体变更"],
    "项目范围管理": ["范围", "WBS", "工作分解", "需求", "确认范围", "范围蔓延", "镀金", "范围基准"],
    "项目进度管理": ["进度", "甘特图", "关键路径", "CPM", "里程碑", "浮动时间", "总时差", "自由时差", "进度压缩", "快速跟进", "赶工"],
    "项目成本管理": ["成本", "挣值", "EVM", "PV", "EV", "AC", "CV", "SV", "CPI", "SPI", "预算", "BAC", "ETC", "EAC", "VAC"],
    "项目质量管理": ["质量", "QA", "QC", "pdca", "六西格玛", "因果图", "控制图", "帕累托", "审计", "核对表"],
    "项目资源管理": ["资源", "团队", "塔克曼", "马斯洛", "赫茨伯格", "责任分配", "RACI", "虚拟团队", "遣散"],
    "项目沟通管理": ["沟通", "沟通模型", "沟通方法", "沟通技术", "报告", "会议", "信息", "反馈"],
    "项目采购管理": ["采购", "合同", "招标", "投标", "卖方", "供方", "工作说明书", "SOW", "激励费", "工料"],
    "项目风险管理": ["风险", "威胁", "机会", "概率影响矩阵", "风险登记册", "定性分析", "定量分析", "应急储备", "权变"],
    "项目相关方管理": ["相关方", "干系人", "利益", "权力利益方格", "参与", "期望", "影响", "相关方登记册"],
}

# 领域概述模板
AREA_DESCRIPTIONS = {
    "项目整体管理": (
        "项目整体管理是识别、定义、组合、统一与协调各项目管理过程组活动的过程，"
        "是项目管理的核心。它包括制定项目章程、制定项目管理计划、指导与管理项目工作、"
        "管理项目知识、监控项目工作、实施整体变更控制、结束项目或阶段等7个过程。"
        "整体管理确保范围、进度、成本、质量等各知识领域协调一致，处理不同目标间的冲突与权衡。"
    ),
    "项目范围管理": (
        "项目范围管理是确保项目做且仅做所需的全部工作，以成功完成项目的过程。"
        "它包括规划范围管理、收集需求、定义范围、创建WBS、确认范围、控制范围等6个过程。"
        "核心概念包括产品范围与项目范围的区分、工作分解结构(WBS)、范围基准、范围蔓延与镀金等。"
    ),
    "项目进度管理": (
        "项目进度管理是管理项目按时完成的过程。"
        "它包括规划进度管理、定义活动、排列活动顺序、估算活动持续时间、制定进度计划、"
        "控制进度等6个过程。核心工具与技术包括关键路径法(CPM)、进度压缩、资源优化技术等。"
    ),
    "项目成本管理": (
        "项目成本管理是确保项目在批准预算内完成的过程。"
        "它包括规划成本管理、估算成本、制定预算、控制成本等4个过程。"
        "核心工具与技术包括挣值管理(EVM)、完工尚需估算(ETC)、完工估算(EAC)等。"
    ),
    "项目质量管理": (
        "项目质量管理是把组织的质量政策应用于规划、管理、控制项目和产品质量要求，"
        "以指导项目并交付预期成果的过程。它包括规划质量管理、管理质量、控制质量等3个过程。"
        "核心概念包括持续改进、预防胜于检查、属性抽样与变量抽样、因果图与控制图等。"
    ),
    "项目资源管理": (
        "项目资源管理是识别、获取和管理所需资源以成功完成项目的过程。"
        "它包括规划资源管理、估算活动资源、获取资源、建设团队、管理团队、控制资源等6个过程。"
        "核心概念包括塔克曼阶梯理论、冲突管理、权力类型、RACI矩阵等。"
    ),
    "项目沟通管理": (
        "项目沟通管理是确保项目信息及时且恰当地规划、收集、生成、发布、存储、检索、"
        "管理、监督和最终处置的过程。它包括规划沟通管理、管理沟通、监督沟通等3个过程。"
        "核心概念包括沟通模型（编码-传递-解码）、沟通方法、沟通渠道数计算等。"
    ),
    "项目采购管理": (
        "项目采购管理是从项目团队外部购买或获取所需产品、服务或成果的过程。"
        "它包括规划采购管理、实施采购、控制采购等3个过程。"
        "核心概念包括合同类型选择（固定总价、成本补偿、工料）、自制或外购分析等。"
    ),
    "项目风险管理": (
        "项目风险管理是规划风险管理、识别、分析、规划应对、实施应对和监督项目风险的过程。"
        "它包括规划风险管理、识别风险、实施定性风险分析、实施定量风险分析、规划风险应对、"
        "实施风险应对、监督风险等7个过程。核心概念包括威胁与机会、概率影响矩阵、应急储备等。"
    ),
    "项目相关方管理": (
        "项目相关方管理是识别影响或受项目影响的人员或组织，分析其期望和影响，"
        "制定策略有效调动其参与项目决策和执行的过程。它包括识别相关方、规划相关方参与、"
        "管理相关方参与、监督相关方参与等4个过程。"
        "核心概念包括权力利益方格、相关方登记册、参与度评估等。"
    ),
}


def _collect_triggers(records, area):
    """从标准记录的 keywords 和领域额外触发词中收集去重触发词。"""
    triggers = set()
    for rec in records:
        for kw in rec.get("keywords", []):
            triggers.add(kw)
    for kw in AREA_EXTRA_TRIGGERS.get(area, []):
        triggers.add(kw)
    return sorted(triggers)


def _collect_process_groups(records):
    """收集该领域涉及的过程组列表（去重、保持顺序）。"""
    seen = set()
    groups = []
    for rec in records:
        pg = rec.get("process_group", "")
        if pg and pg not in seen:
            seen.add(pg)
            groups.append(pg)
    return groups


def _build_yaml_front_matter(data):
    """手动构建 YAML front matter 字符串（不依赖 pyyaml）。"""
    lines = ["---"]
    lines.append(f"name: {data['name']}")
    lines.append(f"area: {data['area']}")
    lines.append(f"area_en: {data['area_en']}")
    lines.append(f"record_count: {data['record_count']}")

    # triggers 列表
    lines.append("triggers:")
    for t in data["triggers"]:
        lines.append(f"  - \"{t}\"")

    # process_groups 列表
    lines.append("process_groups:")
    for pg in data["process_groups"]:
        lines.append(f"  - \"{pg}\"")

    lines.append("---")
    return "\n".join(lines)


def _build_markdown_body(area, area_en, records):
    """构建 Markdown 正文：领域概述 + 标准问答。"""
    desc = AREA_DESCRIPTIONS.get(area, f"{area}（{area_en}）的知识领域。")

    lines = []
    lines.append(f"\n# {area}（{area_en}）\n")
    lines.append(f"{desc}\n")
    lines.append(f"## 标准问答（共 {len(records)} 条）\n")

    for i, rec in enumerate(records, 1):
        rid = rec.get("id", f"Q{i:03d}")
        q = rec.get("question", "")
        a = rec.get("standard_answer", "")
        kws = rec.get("keywords", [])
        pg = rec.get("process_group", "")

        lines.append(f"### {rid}. {q}\n")
        lines.append(f"**标准答案**：{a}\n")
        if kws:
            lines.append(f"**关键词**：{'、'.join(kws)}\n")
        if pg:
            lines.append(f"**过程组**：{pg}\n")
        lines.append("")  # 空行分隔

    return "\n".join(lines)


def generate_skills():
    """读取 pmp_standard/*.json，生成 pmp_skills/*.md。"""
    os.makedirs(SKILLS_DIR, exist_ok=True)

    if not os.path.isdir(STANDARD_DIR):
        print(f"错误：标准目录不存在 {STANDARD_DIR}")
        return

    json_files = sorted(f for f in os.listdir(STANDARD_DIR) if f.endswith(".json"))
    if not json_files:
        print("错误：pmp_standard/ 目录下没有 JSON 文件")
        return

    print(f"开始生成 Skill 文件，共 {len(json_files)} 个知识领域...")

    for fname in json_files:
        fpath = os.path.join(STANDARD_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        area = data.get("area", "")
        area_en = data.get("area_en", "")
        records = data.get("records", [])

        # 生成 skill 文件名（英文小写，下划线分隔）
        skill_name = area_en.lower().replace(" ", "_")
        md_filename = f"{skill_name}.md"
        md_path = os.path.join(SKILLS_DIR, md_filename)

        # 收集触发词和过程组
        triggers = _collect_triggers(records, area)
        process_groups = _collect_process_groups(records)

        # 构建 front matter
        meta = {
            "name": skill_name,
            "area": area,
            "area_en": area_en,
            "record_count": len(records),
            "triggers": triggers,
            "process_groups": process_groups,
        }
        front_matter = _build_yaml_front_matter(meta)

        # 构建 Markdown 正文
        body = _build_markdown_body(area, area_en, records)

        # 写入文件
        content = front_matter + "\n" + body
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"  ✅ {md_filename} ({area}, {len(records)} 条记录, {len(triggers)} 个触发词)")

    print(f"\n完成！Skill 文件已保存到 {SKILLS_DIR}/")


if __name__ == "__main__":
    generate_skills()
