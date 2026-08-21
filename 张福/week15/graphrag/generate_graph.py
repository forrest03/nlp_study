#!/usr/bin/env python3
"""Generate knowledge graph JSON from GUID.md triplets for 2024-2025 Energy Saving and Carbon Reduction Action Plan."""

import json
import os

# ============================================================================
# Entity type definitions
# ============================================================================
ENTITY_TYPES = [
    "政策文件", "政府机构", "年度目标", "能源品类", "工业行业",
    "改造任务", "量化指标", "管理制度", "保障工具", "责任主体",
    "活动载体", "技术类型"
]

# ============================================================================
# Step 1: Define all entities with their types
# ============================================================================
entities = {}

def add_entity(name, etype):
    """Add an entity if not already added."""
    if name not in entities:
        eid = f"E{len(entities) + 1:03d}"
        entities[name] = {"entity_id": eid, "entity_name": name, "entity_type": etype}

# Module 1: Document issuance basics
add_entity("国务院", "政府机构")
add_entity("《2024—2025年节能降碳行动方案》", "政策文件")
add_entity("国发〔2024〕12号", "政策文件")
add_entity("2024年05月23日", "量化指标")
add_entity("2024年05月29日", "量化指标")
add_entity("各省、自治区、直辖市人民政府", "政府机构")
add_entity("国务院各部委", "政府机构")
add_entity("国务院各直属机构", "政府机构")
add_entity("推进双碳、美丽中国、绿色转型核心举措", "政策文件")
add_entity("完成\"十四五\"节能降碳约束性指标", "年度目标")

# Module 2: Guiding ideology
add_entity("习近平新时代中国特色社会主义思想", "技术类型")
add_entity("党的二十大精神", "技术类型")
add_entity("习近平经济思想", "技术类型")
add_entity("习近平生态文明思想", "技术类型")
add_entity("稳中求进", "技术类型")
add_entity("完整准确全面新发展理念", "技术类型")
add_entity("节约优先", "技术类型")
add_entity("完善能源消耗总量和强度调控", "管理制度")
add_entity("控制化石能源消费", "管理制度")
add_entity("强化碳排放强度管理", "管理制度")

# Module 3: Overall targets - 2024
add_entity("2024年节能目标", "年度目标")
add_entity("GDP能耗降幅", "量化指标")
add_entity("2.5%左右", "量化指标")
add_entity("2024年降碳目标", "年度目标")
add_entity("GDP碳排放降幅", "量化指标")
add_entity("3.9%左右", "量化指标")
add_entity("2024年工业目标", "年度目标")
add_entity("规上工业单位增加值能耗降幅", "量化指标")
add_entity("3.5%左右", "量化指标")  # duplicate
add_entity("2024年能源结构目标", "年度目标")
add_entity("非化石能源消费占比", "量化指标")
add_entity("18.9%左右", "量化指标")
add_entity("2024-2025改造总目标", "年度目标")
add_entity("节能量", "量化指标")
add_entity("约5000万吨标准煤", "量化指标")
add_entity("二氧化碳减排量", "量化指标")
add_entity("约1.3亿吨", "量化指标")

# Module 3: 2025 targets
add_entity("2025年能源结构目标", "年度目标")
add_entity("20%左右", "量化指标")
add_entity("2025年电力结构目标", "年度目标")
add_entity("非化石能源发电量占比", "量化指标")
add_entity("39%左右", "量化指标")
add_entity("2025储能装机目标", "年度目标")
add_entity("抽水蓄能装机规模", "量化指标")
add_entity("超6200万千瓦", "量化指标")
add_entity("新型储能装机规模", "量化指标")
add_entity("超4000万千瓦", "量化指标")
add_entity("区域用电要求", "年度目标")
add_entity("一般地区需求响应负荷占比", "量化指标")
add_entity("3%-5%", "量化指标")
add_entity("峰谷差超40%地区需求响应负荷占比", "量化指标")
add_entity("5%以上", "量化指标")

# Module 4: Fossil energy - Coal
add_entity("政策要求", "管理制度")
add_entity("管控煤炭消费", "管理制度")
add_entity("严格合理控制", "管理制度")
add_entity("煤电改造任务", "改造任务")
add_entity("实施改造", "改造任务")
add_entity("节能、灵活、供热三改联动", "改造任务")
add_entity("大气重点区域政策", "管理制度")
add_entity("总量控制、削减非电力用煤", "管理制度")
add_entity("大气重点区域任务", "改造任务")
add_entity("燃煤设施治理", "改造任务")
add_entity("关停整合锅炉、窑炉清洁能源替代", "改造任务")
add_entity("新建煤项目规则", "管理制度")
add_entity("实施替代", "管理制度")
add_entity("煤炭等量/减量替代", "管理制度")
add_entity("产业调控", "管理制度")
add_entity("管控行业", "管理制度")
add_entity("半焦(兰炭)产业规模", "工业行业")
add_entity("2025散煤治理目标", "年度目标")
add_entity("平原散煤治理", "改造任务")
add_entity("基本清零", "改造任务")
add_entity("2025锅炉淘汰目标", "年度目标")
add_entity("淘汰设备", "改造任务")
add_entity("35蒸吨/小时及以下燃煤锅炉", "工业行业")

# Module 4: Oil & Gas
add_entity("石油调控", "能源品类")
add_entity("消费管控", "管理制度")
add_entity("合理调控", "管理制度")
add_entity("油品替代方案", "改造任务")
add_entity("推广燃料", "能源品类")
add_entity("生物液体燃料、可持续航空燃料", "能源品类")
add_entity("油气开发方向", "能源品类")
add_entity("开发资源", "能源品类")
add_entity("页岩油、煤层气、致密油气", "能源品类")
add_entity("天然气使用优先级", "管理制度")
add_entity("优先保障", "管理制度")
add_entity("居民生活、北方清洁取暖", "保障工具")
add_entity("天然气使用限制", "管理制度")
add_entity("自备机组约束", "管理制度")
add_entity("石化外不得新增气电自备机组", "管理制度")

# Module 5: Non-fossil energy
add_entity("新能源基地建设", "改造任务")
add_entity("重点布局", "改造任务")
add_entity("沙漠、戈壁、荒漠大型风光基地", "改造任务")
add_entity("海上能源任务", "改造任务")
add_entity("有序开发", "改造任务")
add_entity("海上风电、海洋能", "能源品类")
add_entity("水电发展策略", "能源品类")
add_entity("有序建设", "改造任务")
add_entity("大型水电基地", "改造任务")
add_entity("核电发展策略", "能源品类")
add_entity("发展路径", "管理制度")
add_entity("安全有序发展", "管理制度")
add_entity("生物质能发展", "能源品类")
add_entity("布局方式", "管理制度")
add_entity("因地制宜开发", "管理制度")
add_entity("氢能发展", "能源品类")
add_entity("推进路径", "管理制度")
add_entity("统筹布局发展", "管理制度")
add_entity("新能源消纳举措", "改造任务")
add_entity("建设通道", "改造任务")
add_entity("风光基地外送特高压通道", "改造任务")
add_entity("电网升级任务", "改造任务")
add_entity("改造对象", "改造任务")
add_entity("配电网提升分布式新能源承载力", "改造任务")
add_entity("新型消纳模式", "技术类型")
add_entity("发展品类", "技术类型")
add_entity("微电网、虚拟电厂、车网互动", "技术类型")
add_entity("高耗能项目约束", "管理制度")
add_entity("2024-2025新建高耗能项目", "管理制度")
add_entity("非化石能源占比≥20%", "量化指标")
add_entity("绿证政策节点", "管理制度")
add_entity("2024年底", "量化指标")
add_entity("实现绿证核发全覆盖", "管理制度")

# Module 6: Steel
add_entity("钢铁管控手段", "工业行业")
add_entity("产能管理", "管理制度")
add_entity("严格产能置换、严控新增", "管理制度")
add_entity("粗钢调控", "工业行业")
add_entity("2024年执行", "年度目标")
add_entity("产量调控", "管理制度")
add_entity("钢铁产品导向", "工业行业")
add_entity("鼓励品类", "工业行业")
add_entity("高性能特种钢", "工业行业")
add_entity("钢铁出口管控", "工业行业")
add_entity("限制品类", "工业行业")
add_entity("低附加值基础钢材出口", "工业行业")
add_entity("短流程炼钢目标", "年度目标")
add_entity("2025电炉钢占比", "量化指标")
add_entity("力争15%", "量化指标")
add_entity("废钢利用目标", "年度目标")
add_entity("2025废钢利用量", "量化指标")
add_entity("3亿吨", "量化指标")
add_entity("钢铁能效目标", "年度目标")
add_entity("2025标杆产能占比", "量化指标")
add_entity("30%", "量化指标")
add_entity("钢铁减排指标", "年度目标")
add_entity("2024-2025节能量", "量化指标")
add_entity("2000万吨标煤", "量化指标")
add_entity("钢铁减排指标_CO2", "年度目标")
add_entity("2024-2025CO₂减排", "量化指标")
add_entity("5300万吨", "量化指标")

# Module 6: Petrochemical
add_entity("石化产能管控", "工业行业")
add_entity("严控新增", "管理制度")
add_entity("炼油、电石、黄磷、磷铵", "工业行业")
add_entity("落后装置淘汰", "改造任务")
add_entity("淘汰设备", "改造任务")
add_entity("200万吨/年以下常减压装置", "工业行业")
add_entity("石化低碳路径", "技术类型")
add_entity("推广技术", "技术类型")
add_entity("可再生能源制氢、绿氢炼化", "技术类型")
add_entity("石化能效目标", "年度目标")
add_entity("超30%", "量化指标")
add_entity("石化减排指标", "年度目标")
add_entity("4000万吨标煤", "量化指标")
add_entity("石化减排指标_CO2", "年度目标")
add_entity("1.1亿吨", "量化指标")

# Module 6: Non-ferrous metals
add_entity("再生金属目标", "年度目标")
add_entity("2025再生金属供应占比", "量化指标")
add_entity("24%以上", "量化指标")
add_entity("铝加工目标", "工业行业")
add_entity("铝水直接合金化比例", "量化指标")
add_entity("90%以上", "量化指标")
add_entity("电解铝绿色目标", "年度目标")
add_entity("2025可再生能源用电占比", "量化指标")
add_entity("25%以上", "量化指标")
add_entity("有色减排指标", "年度目标")
add_entity("500万吨标煤", "量化指标")

# Module 6: Building materials
add_entity("水泥管控手段", "工业行业")
add_entity("产量调节", "管理制度")
add_entity("错峰生产常态化", "管理制度")
add_entity("建材低碳改造", "改造任务")
add_entity("原料替代", "改造任务")
add_entity("尾矿、废渣、工业固废利用", "改造任务")
add_entity("水泥超低排放目标", "年度目标")
add_entity("重点区域熟料改造占比", "量化指标")
add_entity("50%左右", "量化指标")
add_entity("建材减排指标", "年度目标")
add_entity("1000万吨标煤", "量化指标")

# Module 7: Building
add_entity("新建建筑强制标准", "管理制度")
add_entity("城镇新建建筑", "改造任务")
add_entity("全面执行绿色建筑标准", "管理制度")
add_entity("光伏建设目标", "年度目标")
add_entity("新建公建/厂房屋顶光伏覆盖率", "量化指标")
add_entity("力争50%", "量化指标")
add_entity("建筑能源目标", "年度目标")
add_entity("2025城镇建筑可再生替代率", "量化指标")
add_entity("8%", "量化指标")
add_entity("存量改造目标", "年度目标")
add_entity("2025新增节能改造面积", "量化指标")
add_entity("较2023年增长2亿㎡", "量化指标")
add_entity("管网改造目标", "年度目标")
add_entity("2025供热管网热损失降幅", "量化指标")
add_entity("较2020年降低2个百分点", "量化指标")
add_entity("公共建筑管控", "管理制度")
add_entity("管理措施", "管理制度")
add_entity("室内温度管控、智能群控", "管理制度")

# Module 8: Transportation
add_entity("交通基建低碳化", "改造任务")
add_entity("改造内容", "改造任务")
add_entity("岸电、场站光伏、充电设施", "改造任务")
add_entity("运输结构调整", "管理制度")
add_entity("货运转型", "管理制度")
add_entity("公转铁、公转水、多式联运", "管理制度")
add_entity("铁路货运目标", "年度目标")
add_entity("2025货运量增幅", "量化指标")
add_entity("较2020年增长10%", "量化指标")
add_entity("水路货运目标", "年度目标")
add_entity("2025货运量增幅_水路", "量化指标")
add_entity("较2020年增长12%", "量化指标")
add_entity("车辆电动化政策", "管理制度")
add_entity("公共领域车辆", "改造任务")
add_entity("全面电动化", "改造任务")
add_entity("交通减排目标", "年度目标")
add_entity("2025碳排放强度降幅", "量化指标")
add_entity("较2020年降低5%", "量化指标")

# Module 9: Public institutions
add_entity("公共机构考核机制", "管理制度")
add_entity("考核方式", "管理制度")
add_entity("节能目标责任评价考核", "管理制度")
add_entity("2025公建能耗目标", "年度目标")
add_entity("单位建筑面积能耗降幅", "量化指标")
add_entity("较2020年降5%", "量化指标")
add_entity("2025公建碳排放目标", "年度目标")
add_entity("单位建筑面积碳排放降幅", "量化指标")
add_entity("较2020年降7%", "量化指标")
add_entity("2025人均能耗目标", "年度目标")
add_entity("人均综合能耗降幅", "量化指标")
add_entity("较2020年降6%", "量化指标")
add_entity("公建煤炭管控目标", "年度目标")
add_entity("2025煤炭消费占比", "量化指标")
add_entity("降至13%以下", "量化指标")

# Module 10: Equipment update and recycling
add_entity("工业锅炉能效目标", "年度目标")
add_entity("2025热效率提升", "量化指标")
add_entity("较2021年提高5个百分点", "量化指标")
add_entity("高效电机目标", "年度目标")
add_entity("2025高效电机占比提升5个百分点", "量化指标")
add_entity("废旧物资任务", "改造任务")
add_entity("回收品类", "改造任务")
add_entity("工业装备、动力电池、光伏组件", "改造任务")

# Module 11: Management mechanisms
add_entity("节能管控工具", "管理制度")
add_entity("实施考核", "管理制度")
add_entity("项目准入管控", "管理制度")
add_entity("审批手段", "管理制度")
add_entity("固定资产投资节能审查+碳排放评价", "管理制度")
add_entity("监管对象", "管理制度")
add_entity("建立档案", "管理制度")
add_entity("重点用能单位节能管理档案", "管理制度")
add_entity("监察体系建设", "管理制度")
add_entity("搭建层级", "管理制度")
add_entity("省、市、县三级节能监察体系", "管理制度")
add_entity("监察进度目标", "年度目标")
add_entity("2024年底完成60%重点用能单位节能监察", "量化指标")
add_entity("数据支撑体系", "保障工具")
add_entity("建设内容", "保障工具")
add_entity("能耗与碳排放统计快报制度", "保障工具")

# Module 12: Support & guarantee - Standards
add_entity("法制完善任务", "管理制度")
add_entity("修订法律", "管理制度")
add_entity("《节约能源法》", "政策文件")
add_entity("标准分级规则", "管理制度")
add_entity("能效分级", "管理制度")
add_entity("1级标杆前5%、2级先进前20%、3级准入前80%", "管理制度")

# Module 12: Price policy
add_entity("电价约束", "管理制度")
add_entity("禁止行为", "管理制度")
add_entity("对高耗能行业实施电价优惠", "管理制度")
add_entity("电价调节工具", "管理制度")
add_entity("推行制度", "管理制度")
add_entity("高耗能行业阶梯电价", "管理制度")
add_entity("供热改革", "管理制度")
add_entity("推行模式", "管理制度")
add_entity("两部制热价", "管理制度")

# Module 12: Finance
add_entity("资金引导主体", "保障工具")
add_entity("政府资金", "保障工具")
add_entity("支持节能降碳改造、设备更新", "保障工具")
add_entity("金融工具", "保障工具")
add_entity("运用品类", "保障工具")
add_entity("绿色信贷、绿色金融", "保障工具")

# Module 12: Technology
add_entity("技术攻关方向", "技术类型")
add_entity("重点领域", "技术类型")
add_entity("节能降碳关键共性技术", "技术类型")
add_entity("技术推广载体", "技术类型")
add_entity("发布文件", "技术类型")
add_entity("绿色技术推广目录", "技术类型")

# Module 12: Market mechanism
add_entity("节能服务模式", "保障工具")
add_entity("一站式服务", "保障工具")
add_entity("咨询、诊断、融资、托管", "保障工具")
add_entity("交易市场", "保障工具")
add_entity("推进建设", "保障工具")
add_entity("用能权交易、全国碳市场、绿证市场", "保障工具")

# Module 12: Public awareness
add_entity("宣传载体", "活动载体")
add_entity("活动依托", "活动载体")
add_entity("全国节能宣传周、全国低碳日、全国生态日", "活动载体")
add_entity("倡导生活方式", "活动载体")
add_entity("核心导向", "活动载体")
add_entity("简约适度、绿色低碳", "活动载体")

# Module 13: Responsibility
add_entity("统筹主管部门", "政府机构")
add_entity("国家发展改革委", "政府机构")
add_entity("统筹协调、调度考核", "管理制度")
add_entity("碳排放主管部门", "政府机构")
add_entity("生态环境部", "政府机构")
add_entity("碳排放强度目标管理", "管理制度")
add_entity("地方政府责任", "责任主体")
add_entity("各级地方政府", "责任主体")
add_entity("辖区节能降碳总责", "责任主体")
add_entity("政府负责人职责", "责任主体")
add_entity("地方主要负责人", "责任主体")
add_entity("节能降碳第一责任人", "责任主体")
add_entity("市场主体责任", "责任主体")
add_entity("企业", "责任主体")
add_entity("节能降碳主体责任", "责任主体")

# ============================================================================
# Step 2: Define all relations
# ============================================================================
relations = {}

def add_rel(name, desc):
    if name not in relations:
        rid = f"R{len(relations) + 1:03d}"
        relations[name] = {"rel_id": rid, "rel_name": name, "rel_desc": desc}

add_rel("发布文件", "政府机构发布政策文件")
add_rel("发文字号", "政策文件的编号")
add_rel("成文日期", "政策文件的成文日期")
add_rel("发布日期", "政策文件的发布日期")
add_rel("发文对象", "政策文件的发送对象")
add_rel("政策定位", "政策文件的战略定位")
add_rel("制定目的", "政策文件的制定目的")
add_rel("指导理论", "政策文件的指导理论")
add_rel("贯彻会议精神", "政策文件贯彻的会议精神")
add_rel("遵循理论", "政策文件遵循的理论体系")
add_rel("工作总基调", "政策文件的工作总基调")
add_rel("发展理念", "政策文件的发展理念")
add_rel("核心方针", "政策文件的核心方针")
add_rel("调控手段", "政策文件的调控手段")
add_rel("管控重点", "政策文件的管控重点")
add_rel("管理方式", "政策文件的管理方式")
add_rel("GDP能耗降幅", "单位GDP能耗下降幅度")
add_rel("GDP碳排放降幅", "单位GDP碳排放下降幅度")
add_rel("规上工业单位增加值能耗降幅", "规模以上工业单位增加值能耗下降幅度")
add_rel("非化石能源消费占比", "非化石能源消费占比目标")
add_rel("节能量", "节约能源量")
add_rel("二氧化碳减排量", "二氧化碳减排量")
add_rel("非化石能源发电量占比", "非化石能源发电量占比")
add_rel("抽水蓄能装机规模", "抽水蓄能装机规模目标")
add_rel("新型储能装机规模", "新型储能装机规模目标")
add_rel("一般地区需求响应负荷占比", "一般地区需求响应负荷占比")
add_rel("峰谷差超40%地区需求响应负荷占比", "峰谷差超40%地区需求响应负荷占比")
add_rel("管控煤炭消费", "管控煤炭消费行为")
add_rel("实施改造", "实施改造措施")
add_rel("燃煤设施治理", "燃煤设施治理任务")
add_rel("实施替代", "实施替代措施")
add_rel("管控行业", "管控行业范围")
add_rel("平原散煤治理", "平原散煤治理目标")
add_rel("淘汰设备", "淘汰落后设备")
add_rel("消费管控", "消费总量管控")
add_rel("推广燃料", "推广替代燃料")
add_rel("开发资源", "开发能源资源")
add_rel("优先保障", "优先保障供应")
add_rel("自备机组约束", "自备机组约束措施")
add_rel("重点布局", "重点布局区域")
add_rel("有序开发", "有序开发能源")
add_rel("有序建设", "有序建设基地")
add_rel("发展路径", "能源发展路径")
add_rel("因地制宜开发", "因地制宜开发能源")
add_rel("推进路径", "推进发展路径")
add_rel("建设通道", "建设外送通道")
add_rel("改造对象", "改造的对象")
add_rel("发展品类", "新型消纳模式发展品类")
add_rel("产能管理", "产能管理措施")
add_rel("产量调控", "产量调控措施")
add_rel("鼓励品类", "鼓励发展的品类")
add_rel("限制品类", "限制出口的品类")
add_rel("2025电炉钢占比", "2025年电炉钢占比目标")
add_rel("2025废钢利用量", "2025年废钢利用量目标")
add_rel("2025标杆产能占比", "2025年标杆产能占比")
add_rel("2024-2025CO₂减排", "2024-2025年二氧化碳减排量")
add_rel("严控新增", "严控新增产能")
add_rel("推广技术", "推广先进技术")
add_rel("2025再生金属供应占比", "2025年再生金属供应占比")
add_rel("铝水直接合金化比例", "铝水直接合金化比例")
add_rel("2025可再生能源用电占比", "2025年可再生能源用电占比")
add_rel("原料替代", "原料替代措施")
add_rel("水泥超低排放目标", "水泥超低排放目标")
add_rel("全面执行绿色建筑标准", "全面执行绿色建筑标准")
add_rel("力争50%", "覆盖率力争50%")
add_rel("2025城镇建筑可再生替代率", "2025年城镇建筑可再生替代率")
add_rel("存量改造目标", "存量改造目标")
add_rel("管网改造目标", "管网改造目标")
add_rel("室内温度管控、智能群控", "公共建筑管控措施")
add_rel("改造内容", "改造的内容")
add_rel("货运转型", "货运结构转型")
add_rel("2025货运量增幅", "2025年货运量增幅")
add_rel("全面电动化", "车辆全面电动化")
add_rel("交通减排目标", "交通减排目标")
add_rel("考核方式", "考核的方式")
add_rel("单位建筑面积能耗降幅", "单位建筑面积能耗降幅")
add_rel("单位建筑面积碳排放降幅", "单位建筑面积碳排放降幅")
add_rel("人均综合能耗降幅", "人均综合能耗降幅")
add_rel("2025煤炭消费占比", "2025年煤炭消费占比")
add_rel("工业锅炉能效目标", "工业锅炉能效目标")
add_rel("高效电机目标", "高效电机目标")
add_rel("回收品类", "回收的品类")
add_rel("实施考核", "实施考核制度")
add_rel("审批手段", "项目审批手段")
add_rel("建立档案", "建立管理档案")
add_rel("搭建层级", "监察体系层级")
add_rel("法制完善任务", "法制完善任务")
add_rel("修订法律", "修订相关法律")
add_rel("能效分级", "能效等级分级")
add_rel("禁止行为", "禁止的行为")
add_rel("推行制度", "推行的制度")
add_rel("推行模式", "推行的模式")
add_rel("资金引导", "资金引导方向")
add_rel("运用品类", "金融工具运用品类")
add_rel("重点领域", "技术攻关重点领域")
add_rel("发布文件", "发布技术推广文件")
add_rel("一站式服务", "一站式服务模式")
add_rel("推进建设", "推进市场建设")
add_rel("活动依托", "活动依托载体")
add_rel("核心导向", "倡导生活方式核心导向")
add_rel("统筹协调、调度考核", "统筹协调与调度考核职责")
add_rel("碳排放强度目标管理", "碳排放强度目标管理职责")
add_rel("辖区节能降碳总责", "辖区节能降碳总责")
add_rel("节能降碳第一责任人", "节能降碳第一责任人职责")
add_rel("节能降碳主体责任", "节能降碳主体责任")

# ============================================================================
# Step 3: Define all triples from GUID.md
# ============================================================================
triples_raw = []

# Module 1: Document issuance
triples_raw.append(("国务院", "发布文件", "《2024—2025年节能降碳行动方案》"))
triples_raw.append(("《2024—2025年节能降碳行动方案》", "发文字号", "国发〔2024〕12号"))
triples_raw.append(("《2024—2025年节能降碳行动方案》", "成文日期", "2024年05月23日"))
triples_raw.append(("《2024—2025年节能降碳行动方案》", "发布日期", "2024年05月29日"))
triples_raw.append(("《2024—2025年节能降碳行动方案》", "发文对象", "各省、自治区、直辖市人民政府"))
triples_raw.append(("《2024—2025年节能降碳行动方案》", "发文对象", "国务院各部委"))
triples_raw.append(("《2024—2025年节能降碳行动方案》", "发文对象", "国务院各直属机构"))
triples_raw.append(("《2024—2025年节能降碳行动方案》", "政策定位", "推进双碳、美丽中国、绿色转型核心举措"))
triples_raw.append(("《2024—2025年节能降碳行动方案》", "制定目的", "完成\"十四五\"节能降碳约束性指标"))

# Module 2: Guiding ideology
triples_raw.append(("《2024—2025年节能降碳行动方案》", "指导理论", "习近平新时代中国特色社会主义思想"))
triples_raw.append(("《2024—2025年节能降碳行动方案》", "贯彻会议精神", "党的二十大精神"))
triples_raw.append(("《2024—2025年节能降碳行动方案》", "遵循理论", "习近平经济思想"))
triples_raw.append(("《2024—2025年节能降碳行动方案》", "遵循理论", "习近平生态文明思想"))
triples_raw.append(("《2024—2025年节能降碳行动方案》", "工作总基调", "稳中求进"))
triples_raw.append(("《2024—2025年节能降碳行动方案》", "发展理念", "完整准确全面新发展理念"))
triples_raw.append(("《2024—2025年节能降碳行动方案》", "核心方针", "节约优先"))
triples_raw.append(("《2024—2025年节能降碳行动方案》", "调控手段", "完善能源消耗总量和强度调控"))
triples_raw.append(("《2024—2025年节能降碳行动方案》", "管控重点", "控制化石能源消费"))
triples_raw.append(("《2024—2025年节能降碳行动方案》", "管理方式", "强化碳排放强度管理"))

# Module 3.1: 2024 targets
triples_raw.append(("2024年节能目标", "GDP能耗降幅", "2.5%左右"))
triples_raw.append(("2024年降碳目标", "GDP碳排放降幅", "3.9%左右"))
triples_raw.append(("2024年工业目标", "规上工业单位增加值能耗降幅", "3.5%左右"))
triples_raw.append(("2024年能源结构目标", "非化石能源消费占比", "18.9%左右"))
triples_raw.append(("2024-2025改造总目标", "节能量", "约5000万吨标准煤"))
triples_raw.append(("2024-2025改造总目标", "二氧化碳减排量", "约1.3亿吨"))

# Module 3.2: 2025 targets
triples_raw.append(("2025年能源结构目标", "非化石能源消费占比", "20%左右"))
triples_raw.append(("2025年电力结构目标", "非化石能源发电量占比", "39%左右"))
triples_raw.append(("2025储能装机目标", "抽水蓄能装机规模", "超6200万千瓦"))
triples_raw.append(("2025储能装机目标", "新型储能装机规模", "超4000万千瓦"))
triples_raw.append(("区域用电要求", "一般地区需求响应负荷占比", "3%-5%"))
triples_raw.append(("区域用电要求", "峰谷差超40%地区需求响应负荷占比", "5%以上"))

# Module 4.1: Coal
triples_raw.append(("政策要求", "管控煤炭消费", "严格合理控制"))
triples_raw.append(("煤电改造任务", "实施改造", "节能、灵活、供热三改联动"))
triples_raw.append(("大气重点区域政策", "燃煤设施治理", "总量控制、削减非电力用煤"))
triples_raw.append(("大气重点区域任务", "燃煤设施治理", "关停整合锅炉、窑炉清洁能源替代"))
triples_raw.append(("新建煤项目规则", "实施替代", "煤炭等量/减量替代"))
triples_raw.append(("产业调控", "管控行业", "半焦(兰炭)产业规模"))
triples_raw.append(("2025散煤治理目标", "平原散煤治理", "基本清零"))
triples_raw.append(("2025锅炉淘汰目标", "淘汰设备", "35蒸吨/小时及以下燃煤锅炉"))

# Module 4.2: Oil & Gas
triples_raw.append(("石油调控", "消费管控", "合理调控"))
triples_raw.append(("油品替代方案", "推广燃料", "生物液体燃料、可持续航空燃料"))
triples_raw.append(("油气开发方向", "开发资源", "页岩油、煤层气、致密油气"))
triples_raw.append(("天然气使用优先级", "优先保障", "居民生活、北方清洁取暖"))
triples_raw.append(("天然气使用限制", "自备机组约束", "石化外不得新增气电自备机组"))

# Module 5: Non-fossil energy
triples_raw.append(("新能源基地建设", "重点布局", "沙漠、戈壁、荒漠大型风光基地"))
triples_raw.append(("海上能源任务", "有序开发", "海上风电、海洋能"))
triples_raw.append(("水电发展策略", "有序建设", "大型水电基地"))
triples_raw.append(("核电发展策略", "发展路径", "安全有序发展"))
triples_raw.append(("生物质能发展", "因地制宜开发", "因地制宜开发"))
triples_raw.append(("氢能发展", "推进路径", "统筹布局发展"))
triples_raw.append(("新能源消纳举措", "建设通道", "风光基地外送特高压通道"))
triples_raw.append(("电网升级任务", "改造对象", "配电网提升分布式新能源承载力"))
triples_raw.append(("新型消纳模式", "发展品类", "微电网、虚拟电厂、车网互动"))
triples_raw.append(("高耗能项目约束", "管控重点", "2024-2025新建高耗能项目"))
triples_raw.append(("高耗能项目约束", "非化石能源消费占比", "非化石能源占比≥20%"))
triples_raw.append(("绿证政策节点", "发布文件", "2024年底"))
triples_raw.append(("绿证政策节点", "推进建设", "实现绿证核发全覆盖"))

# Module 6.1: Steel
triples_raw.append(("钢铁管控手段", "产能管理", "严格产能置换、严控新增"))
triples_raw.append(("粗钢调控", "产量调控", "2024年执行"))
triples_raw.append(("钢铁产品导向", "鼓励品类", "高性能特种钢"))
triples_raw.append(("钢铁出口管控", "限制品类", "低附加值基础钢材出口"))
triples_raw.append(("短流程炼钢目标", "2025电炉钢占比", "力争15%"))
triples_raw.append(("废钢利用目标", "2025废钢利用量", "3亿吨"))
triples_raw.append(("钢铁能效目标", "2025标杆产能占比", "30%"))
triples_raw.append(("钢铁减排指标", "节能量", "2000万吨标煤"))
triples_raw.append(("钢铁减排指标_CO2", "2024-2025CO₂减排", "5300万吨"))

# Module 6.2: Petrochemical
triples_raw.append(("石化产能管控", "严控新增", "炼油、电石、黄磷、磷铵"))
triples_raw.append(("落后装置淘汰", "淘汰设备", "200万吨/年以下常减压装置"))
triples_raw.append(("石化低碳路径", "推广技术", "可再生能源制氢、绿氢炼化"))
triples_raw.append(("石化能效目标", "2025标杆产能占比", "超30%"))
triples_raw.append(("石化减排指标", "节能量", "4000万吨标煤"))
triples_raw.append(("石化减排指标_CO2", "2024-2025CO₂减排", "1.1亿吨"))

# Module 6.3: Non-ferrous
triples_raw.append(("再生金属目标", "2025再生金属供应占比", "24%以上"))
triples_raw.append(("铝加工目标", "铝水直接合金化比例", "90%以上"))
triples_raw.append(("电解铝绿色目标", "2025可再生能源用电占比", "25%以上"))
triples_raw.append(("有色减排指标", "节能量", "500万吨标煤"))

# Module 6.4: Building materials
triples_raw.append(("水泥管控手段", "产量调控", "错峰生产常态化"))
triples_raw.append(("建材低碳改造", "原料替代", "尾矿、废渣、工业固废利用"))
triples_raw.append(("水泥超低排放目标", "水泥超低排放目标", "重点区域熟料改造占比"))
triples_raw.append(("水泥超低排放目标", "量化指标", "50%左右"))
triples_raw.append(("建材减排指标", "节能量", "1000万吨标煤"))

# Module 7: Building
triples_raw.append(("新建建筑强制标准", "管控重点", "城镇新建建筑"))
triples_raw.append(("城镇新建建筑", "全面执行绿色建筑标准", "全面执行绿色建筑标准"))
triples_raw.append(("光伏建设目标", "量化指标", "新建公建/厂房屋顶光伏覆盖率"))
triples_raw.append(("新建公建/厂房屋顶光伏覆盖率", "力争50%", "力争50%"))
triples_raw.append(("建筑能源目标", "量化指标", "2025城镇建筑可再生替代率"))
triples_raw.append(("2025城镇建筑可再生替代率", "量化指标", "8%"))
triples_raw.append(("存量改造目标", "量化指标", "2025新增节能改造面积"))
triples_raw.append(("2025新增节能改造面积", "量化指标", "较2023年增长2亿㎡"))
triples_raw.append(("管网改造目标", "量化指标", "2025供热管网热损失降幅"))
triples_raw.append(("2025供热管网热损失降幅", "量化指标", "较2020年降低2个百分点"))
triples_raw.append(("公共建筑管控", "管理方式", "室内温度管控、智能群控"))

# Module 8: Transportation
triples_raw.append(("交通基建低碳化", "改造内容", "岸电、场站光伏、充电设施"))
triples_raw.append(("运输结构调整", "货运转型", "公转铁、公转水、多式联运"))
triples_raw.append(("铁路货运目标", "2025货运量增幅", "较2020年增长10%"))
triples_raw.append(("水路货运目标", "2025货运量增幅", "较2020年增长12%"))
triples_raw.append(("车辆电动化政策", "管控重点", "公共领域车辆"))
triples_raw.append(("公共领域车辆", "全面电动化", "全面电动化"))
triples_raw.append(("交通减排目标", "量化指标", "2025碳排放强度降幅"))
triples_raw.append(("2025碳排放强度降幅", "量化指标", "较2020年降低5%"))

# Module 9: Public institutions
triples_raw.append(("公共机构考核机制", "考核方式", "节能目标责任评价考核"))
triples_raw.append(("2025公建能耗目标", "单位建筑面积能耗降幅", "较2020年降5%"))
triples_raw.append(("2025公建碳排放目标", "单位建筑面积碳排放降幅", "较2020年降7%"))
triples_raw.append(("2025人均能耗目标", "人均综合能耗降幅", "较2020年降6%"))
triples_raw.append(("公建煤炭管控目标", "2025煤炭消费占比", "降至13%以下"))

# Module 10: Equipment update & recycling
triples_raw.append(("工业锅炉能效目标", "量化指标", "2025热效率提升"))
triples_raw.append(("2025热效率提升", "量化指标", "较2021年提高5个百分点"))
triples_raw.append(("高效电机目标", "量化指标", "2025高效电机占比提升5个百分点"))
triples_raw.append(("废旧物资任务", "回收品类", "工业装备、动力电池、光伏组件"))

# Module 11: Management mechanisms
triples_raw.append(("节能管控工具", "实施考核", "节能目标责任评价考核"))
triples_raw.append(("项目准入管控", "审批手段", "固定资产投资节能审查+碳排放评价"))
triples_raw.append(("监管对象", "建立档案", "重点用能单位节能管理档案"))
triples_raw.append(("监察体系建设", "搭建层级", "省、市、县三级节能监察体系"))
triples_raw.append(("监察进度目标", "量化指标", "2024年底完成60%重点用能单位节能监察"))
triples_raw.append(("数据支撑体系", "建设内容", "能耗与碳排放统计快报制度"))

# Module 12.1: Standards
triples_raw.append(("法制完善任务", "修订法律", "《节约能源法》"))
triples_raw.append(("标准分级规则", "能效分级", "1级标杆前5%、2级先进前20%、3级准入前80%"))

# Module 12.2: Price policy
triples_raw.append(("电价约束", "禁止行为", "对高耗能行业实施电价优惠"))
triples_raw.append(("电价调节工具", "推行制度", "高耗能行业阶梯电价"))
triples_raw.append(("供热改革", "推行模式", "两部制热价"))

# Module 12.3: Finance
triples_raw.append(("资金引导主体", "资金引导", "政府资金"))
triples_raw.append(("政府资金", "管控重点", "支持节能降碳改造、设备更新"))
triples_raw.append(("金融工具", "运用品类", "绿色信贷、绿色金融"))

# Module 12.4: Technology
triples_raw.append(("技术攻关方向", "重点领域", "节能降碳关键共性技术"))
triples_raw.append(("技术推广载体", "发布文件", "绿色技术推广目录"))

# Module 12.5: Market mechanism
triples_raw.append(("节能服务模式", "一站式服务", "咨询、诊断、融资、托管"))
triples_raw.append(("交易市场", "推进建设", "用能权交易、全国碳市场、绿证市场"))

# Module 12.6: Public awareness
triples_raw.append(("宣传载体", "活动依托", "全国节能宣传周、全国低碳日、全国生态日"))
triples_raw.append(("倡导生活方式", "核心导向", "简约适度、绿色低碳"))

# Module 13: Responsibility
triples_raw.append(("统筹主管部门", "统筹协调、调度考核", "国家发展改革委"))
triples_raw.append(("碳排放主管部门", "碳排放强度目标管理", "生态环境部"))
triples_raw.append(("地方政府责任", "辖区节能降碳总责", "各级地方政府"))
triples_raw.append(("政府负责人职责", "节能降碳第一责任人", "地方主要负责人"))
triples_raw.append(("市场主体责任", "节能降碳主体责任", "企业"))

# ============================================================================
# Step 4: Build the output structures
# ============================================================================
# Ensure all entities referenced in triples are registered
for sub, rel, obj in triples_raw:
    if sub not in entities:
        add_entity(sub, "技术类型")
    if obj not in entities:
        add_entity(obj, "技术类型")
    if rel not in relations:
        add_rel(rel, rel)

# Build entity list sorted by entity_id
entity_list = sorted(entities.values(), key=lambda x: x["entity_id"])

# Build relation list sorted by rel_id
relation_list = sorted(relations.values(), key=lambda x: x["rel_id"])

# Build triple list
triple_list = []
for sub, rel, obj in triples_raw:
    triple_list.append({
        "sub_id": entities[sub]["entity_id"],
        "rel_id": relations[rel]["rel_id"],
        "obj_id": entities[obj]["entity_id"]
    })

# ============================================================================
# Step 5: Write output
# ============================================================================
output = {
    "graph_name": "2024—2025年节能降碳行动方案知识图谱",
    "entity_list": entity_list,
    "relation_list": relation_list,
    "triple_list": triple_list
}

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "graphs")
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "2024-2025年节能降碳行动方案知识图谱.json")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Graph JSON generated successfully: {output_path}")
print(f"  Entities: {len(entity_list)}")
print(f"  Relations: {len(relation_list)}")
print(f"  Triples: {len(triple_list)}")