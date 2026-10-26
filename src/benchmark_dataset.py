import json
from pathlib import Path

DOMAINS = [
    ("新能源汽车", ["销量与渗透率", "厂商竞争格局", "政策与补贴"]),
    ("咖啡", ["市场规模", "品牌与价格竞争", "消费人群与渠道"]),
    ("扫地机器人", ["销量与价格带", "厂商与产品竞争", "导航及清洁技术"]),
    ("宠物经济", ["市场规模", "食品与医疗细分", "消费人群变化"]),
    ("养老服务", ["人口与需求", "服务供给与商业模式", "政策与支付能力"]),
    ("储能", ["装机与需求", "技术路线与成本", "政策与安全风险"]),
    ("低空经济", ["应用与市场需求", "企业与产业链", "监管与基础设施"]),
    ("AI办公软件", ["企业需求与付费", "产品与竞争", "数据安全及落地障碍"]),
    ("即时零售", ["规模与订单需求", "平台与商户竞争", "履约成本与盈利"]),
    ("跨境电商", ["市场与出口规模", "平台及卖家竞争", "关税及合规风险"]),
]
ANGLES = [
    ("概况", "调研中国{domain}行业2025年的发展概况，分别覆盖{facets}。区分实际数据与预测，给出出处。"),
    ("跨期", "比较中国{domain}行业2024与2025年的变化，从{facets}分析。明确年份、单位和无法核实之处。"),
    ("细分", "分析中国{domain}行业主要细分场景的差异，按{facets}组织。不要把个别案例外推为全国结论。"),
    ("决策", "为计划进入中国{domain}行业的企业做调研，按{facets}给出机会、约束和待验证假设，区分证据与建议。"),
    ("风险", "核对中国{domain}行业增长叙事中的风险，分别调查{facets}，说明证据冲突、数据口径及不确定性。"),
]


def build_dataset():
    tasks = []
    for domain, facets in DOMAINS:
        for angle, template in ANGLES:
            tasks.append({"id": f"R{len(tasks)+1:03d}", "domain": domain,
                          "angle": angle, "question": template.format(
                              domain=domain, facets="、".join(facets)),
                          "required_facets": facets,
                          "fixed_topics": [f"中国{domain}行业2024-2025年 {f} 实际数据 来源"
                                           for f in facets]})
    return {"version": "research50-v1", "authorship": "local_authored",
            "limitations": "任务与维度由作者构建，非权威行业基准，无金标事实答案；Judge只评估保存证据支持性。",
            "tasks": tasks}


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[1] / "data/benchmark_50.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_dataset(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(path.name, "50 tasks")
