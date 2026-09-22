"""Bind source-checked facts/figures to original elements, without consulting retrieval results."""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from docqa.evaluation import ExperimentRequest

CORPUS = ROOT / "data/corpus"
manifest = json.loads((ROOT / "datasets/public_sources.json").read_text())
imports = json.loads((CORPUS / "import-report.json").read_text())
imported = {d["sha256"]: d for d in imports["documents"]}
questions = []


def compact(text):
    return re.sub(r"\s+", "", text)


def elements(source):
    doc = imported[source["sha256"]]
    return json.loads((CORPUS / (doc["document_id"] + "-elements.json")).read_text())


def add(source, category, question, answer, gold, fact_group, split="test", points=None):
    refs = []
    for e in gold:
        refs.extend(
            {**s, "document_id": e["document_id"], "document_sha256": source["sha256"]} for s in e["sources"]
        )
    questions.append(
        {
            "id": f"public-{len(questions) + 1:03}",
            "question": question,
            "category": category,
            "reference_answer": answer,
            "relevant_element_ids": [e["id"] for e in gold],
            "reference_sources": refs,
            "scoring_points": points or [answer],
            "fact_group": fact_group,
            "split": split,
            "answerable": category != "unanswerable",
        }
    )


monthly = [s for s in manifest["documents"] if s["title"].endswith("全国城市空气质量报告")]
cnnic = next(s for s in manifest["documents"] if s["title"].startswith("第55"))
for i, source in enumerate(monthly):
    es = elements(source)
    candidates = [
        e
        for e in es
        if e["kind"] == "text" and "全国" in e["text"] and "平均空气质量优良天数" in compact(e["text"])
    ]
    original = min(candidates, key=lambda e: e["sources"][0]["page"])
    text = compact(original["text"])
    ratio = re.search(r"平均空气质量优良天数比例为([0-9.]+)%", text)[1]
    pm = re.search(r"PM2\.5平均浓度为([0-9.]+)", text)[1]
    with pymupdf.open(CORPUS / source["local_path"]) as pdf:
        raw = compact("".join(p.get_text() for p in pdf))
    if len(raw) > 100:
        assert re.search(r"平均空气质量优良天数比例为([0-9.]+)%", raw)[1] == ratio
        assert re.search(r"PM2\.5平均浓度为([0-9.]+)", raw)[1] == pm
    else:
        # This scan was separately inspected on PDF page 3; no inference from a model answer.
        assert source["title"] == "2025年12月全国城市空气质量报告" and (ratio, pm) == ("84.8", "44.6")
    date = source["title"].split("全国")[0]
    split = "dev" if i < 5 else "test"
    add(
        source,
        "text",
        f"据《{source['title']}》，当月全国地级及以上城市的平均空气质量优良天数比例是多少？",
        ratio + "%",
        [original],
        f"{date}-national-good-days",
        split,
    )
    add(
        source,
        "text",
        f"《{source['title']}》给出的当月全国城市PM2.5平均浓度是多少？请注明单位。",
        pm + " μg/m³",
        [original],
        f"{date}-national-pm25",
        split,
    )

cn = elements(cnnic)
for page, anchor, question, answer, fact in [
    (
        9,
        "64K",
        "第55次互联网统计报告回顾：中国全功能接入国际互联网的专线在哪一天开通，带宽是多少？",
        "1994年4月20日；64K。",
        "china-first-internet-connection",
    ),
    (
        8,
        "2.49",
        "第55次互联网统计报告中，截至2024年12月生成式人工智能产品的用户规模和占整体人口比例分别是多少？",
        "2.49亿人；17.7%。",
        "genai-total-users-2024",
    ),
]:
    gold = next(
        e for e in cn if e["kind"] == "text" and e["sources"][0]["page"] == page and anchor in e["text"]
    )
    add(cnnic, "text", question, answer, [gold], fact)

# Reference table values were checked against the printed standard tables, not predictions.
standards = [
    ("SO2", "年平均", "一级和二级", "20、60 μg/m³"),
    ("NO2", "年平均", "一级和二级", "40、40 μg/m³"),
    ("CO", "24小时平均", "一级和二级", "4、4 mg/m³"),
    ("O3", "8小时平均", "一级和二级", "100、160 μg/m³"),
    ("PM10", "年平均", "一级和二级", "40、70 μg/m³"),
    ("PM2.5", "年平均", "一级和二级", "15、35 μg/m³"),
    ("SO2", "24小时平均", "一级和二级", "50、150 μg/m³"),
    ("SO2", "1小时平均", "一级和二级", "150、500 μg/m³"),
    ("NO2", "24小时平均", "一级和二级", "80、80 μg/m³"),
    ("NO2", "1小时平均", "一级和二级", "200、200 μg/m³"),
    ("CO", "1小时平均", "一级和二级", "10、10 mg/m³"),
    ("O3", "1小时平均", "一级和二级", "160、200 μg/m³"),
    ("PM10", "24小时平均", "一级和二级", "50、150 μg/m³"),
    ("PM2.5", "24小时平均", "一级和二级", "35、75 μg/m³"),
    ("PM10", "年平均", "二级与一级限值之比", "70/40 = 1.75倍"),
    ("SO2", "年平均", "二级减一级的差值", "60-20 = 40 μg/m³"),
    ("PM2.5", "24小时平均", "二级减一级的差值", "75-35 = 40 μg/m³"),
    ("NO2", "1小时平均", "二级与一级限值之比", "200/200 = 1倍"),
    ("CO", "24小时平均和1小时平均", "二级限值分别", "4、10 mg/m³"),
]
for source, (pollutant, duration, level, answer) in zip(monthly, standards, strict=True):
    candidates = [
        e
        for e in elements(source)
        if e["kind"] == "table" and "年平均" in e["text"] and "浓度限值" in e["text"]
    ]
    assert len(candidates) == 1, source["title"]
    add(
        source,
        "table",
        f"请查《{source['title']}》的“环境空气污染物基本项目浓度限值”表：{pollutant}的{duration}{level}是多少？",
        answer,
        candidates,
        f"standard-{pollutant}-{duration}-{level}",
    )

for page, anchor, question, answer, fact in [
    (17, "分类域名数", "表2中.COM域名的准确数量是多少个？", "7,047,974个", "domain-com"),
    (17, "分类域名数", "表2中“.中国”域名的准确数量是多少个？", "165,265个", "domain-china"),
    (18, ".GOV.CN", "表3中.GOV.CN域名的数量是多少？", "12,608个", "domain-gov"),
    (18, ".GOV.CN", "表3中.EDU.CN域名的数量是多少？", "6,857个", "domain-edu"),
    (
        33,
        "即时通信",
        "表5中即时通信在2024年12月的用户规模和网民使用率分别是多少？",
        "108,133万人；97.6%。",
        "im-users-2024",
    ),
    (
        33,
        "即时通信",
        "表5中的短视频用户规模增长率是多少，是否为负增长？",
        "-1.3%；是负增长。",
        "short-video-growth",
    ),
]:
    gold = next(
        e for e in cn if e["kind"] == "table" and e["sources"][0]["page"] == page and anchor in e["text"]
    )
    add(cnnic, "table", "根据第55次《中国互联网络发展状况统计报告》，" + question, answer, [gold], fact)

figures = [
    (
        1,
        16,
        "图1所示的IPv6地址数量从2020年12月至2024年12月呈什么变化趋势，最后一期是多少？",
        "逐年增加；2024年12月为69,148块/32。",
    ),
    (2, 16, "图2中2021年12月的IPv6活跃用户数是多少？", "5.35亿。"),
    (4, 18, "读取图4，2022年12月的5G基站数量是多少？", "231.2万个。"),
    (5, 19, "根据图5，2024年11月比2020年12月增加了多少互联网宽带接入端口？", "11.99-9.46 = 2.53亿个。"),
    (6, 19, "图6中2022年12月的光缆线路总长度是多少？", "5,958万公里。"),
    (7, 20, "根据图7，从2023年12月到2024年12月网站数量增加了多少？", "446-388 = 58万个。"),
    (8, 20, "图8中2021年12月的网页数量是多少？", "3,350亿个。"),
    (10, 23, "图10中2021年12月的网民规模和互联网普及率分别是多少？", "103,195万人；73.0%。"),
    (11, 24, "图11中2022年12月的手机网民规模及其占网民比例分别是多少？", "106,510万人；99.8%。"),
    (
        12,
        25,
        "图12显示，农村网民占比从2023年12月至2024年12月改变了多少个百分点？",
        "从29.8%降至28.2%，下降1.6个百分点。",
    ),
    (13, 27, "图13列出的不上网生活不便中，占比最低的是哪项，比例是多少？", "很难买到火车票、飞机票；5.0%。"),
    (
        14,
        27,
        "图14中因没有电脑等上网设备而不上网的比例，比不需要或不感兴趣高多少个百分点？",
        "13.0%-6.1% = 6.9个百分点。",
    ),
    (
        15,
        28,
        "图15中促进非网民上网的前两项因素及其比例是什么？",
        "方便与家人或亲属沟通联系18.7%；提供可以无障碍使用的上网设备18.4%。",
    ),
    (16, 28, "按图16，男性网民占比比女性高多少个百分点？", "51.1%-48.9% = 2.2个百分点。"),
    (17, 29, "按图17的两个年龄分组相加，50岁及以上网民合计占比是多少？", "20.0%+14.1% = 34.1%。"),
    (18, 29, "图18中排除手机后，使用比例最高的上网设备是哪一种，比例是多少？", "台式电脑；36.2%。"),
    (
        19,
        30,
        "图19中网民人均每周上网时长最高和最低的年份分别是哪年，差多少小时？",
        "最高2024年28.7小时，最低2023年26.1小时；差2.6小时。",
    ),
    (
        21,
        32,
        "图21中使用编程语言编写计算机程序的熟练掌握、基本掌握比例及合计是多少？",
        "3.6%、4.9%，合计8.5%。",
    ),
    (
        25,
        38,
        "图25中关注度最高和第二高的议题分别是什么，比例差是多少？",
        "社会新闻63.5%；养老、就业、教育等资讯50.7%；差12.8个百分点。",
    ),
    (
        35,
        50,
        "图35中生成式人工智能使用率最高与最低的是哪个年龄段，比例相差多少？",
        "20-29岁最高41.5%，50-59岁最低10.7%；相差30.8个百分点。",
    ),
]
for index, (figure, page, question, answer) in enumerate(figures):
    candidates = [
        e
        for e in cn
        if e["kind"] == "image"
        and e["sources"][0]["page"] == page
        and re.match(rf"图\s*{figure}(?!\d)", e["text"])
    ]
    assert len(candidates) == 1, (figure, [e["id"] for e in candidates])
    add(
        cnnic,
        "chart",
        "根据第55次《中国互联网络发展状况统计报告》，" + question,
        answer,
        candidates,
        f"cnnic55-figure-{figure}",
        "dev" if index < 8 else "test",
    )

unknowns = [
    "2025年12月全国城市空气质量报告有没有给出2027年12月全国PM2.5平均浓度的预测值？预测值是多少？",
    "2025年11月全国城市空气质量报告中，全国监测站购买PM2.5分析仪的采购总金额是多少？",
    "2024年12月全国城市空气质量报告是否披露了每个监测站负责人名单？具体名单是什么？",
    "2024年11月全国城市空气质量报告给出的全国空气监测仪器平均使用寿命是多少年？",
    "2024年10月全国城市空气质量报告给出了北京某一个家庭室内PM2.5的实测值吗？是多少？",
    "2024年9月全国城市空气质量报告是否披露了监测设备的具体品牌和型号？分别是什么？",
    "2024年8月全国城市空气质量报告中，为改善空气质量新增的财政投入总额是多少？",
    "2023年12月全国城市空气质量报告中，用于预测污染物浓度的神经网络有多少参数？",
    "2023年6月全国城市空气质量报告是否列出了全部城市每小时的原始监测序列？完整序列是什么？",
    "2025年10月全国城市空气质量报告中，监测数据存储数据库使用了哪种数据库软件？",
    "2025年9月全国城市空气质量报告中，编制该报告的人工工时总计是多少？",
    "2025年8月全国城市空气质量报告给出了2027年环境治理项目的逐项预算吗？金额是多少？",
    "第55次互联网统计报告是否公布了每一名受访者的原始逐题答卷？请列出全部答卷。",
    "第55次互联网统计报告是否预测了2030年我国网民规模的确定数值？具体是多少？",
    "第55次互联网统计报告中，受访者使用的生成式人工智能模型训练GPU数量分别是多少？",
]
for i, question in enumerate(unknowns):
    add(
        None,
        "unanswerable",
        question,
        "当前文档中没有足够依据。",
        [],
        f"unsupported-{i + 1}",
        "dev" if i < 2 else "test",
        points=["明确说明文档未提供该信息；不编造数值、名单或序列。"],
    )

counts = Counter(q["category"] for q in questions)
assert counts == {"text": 40, "table": 25, "chart": 20, "unanswerable": 15}, counts
assert Counter(q["split"] for q in questions) == {"dev": 20, "test": 80}
request = ExperimentRequest(
    questions=questions, document_ids=[imported[d["sha256"]]["document_id"] for d in manifest["documents"]]
)
output = {
    "name": "公开中文报告100题 v1",
    "document_ids": request.document_ids,
    "annotation_method": "从原始PDF文本、印刷表格和已查看的图页提取事实，绑定原始元素；未用检索结果或模型回答生成金标准。",
    "limitations": [
        "语料以空气质量月报为主，题型分布不代表通用领域。",
        "表格问题部分针对重复标准表；未跨开发/测试划分相同事实组。",
        "图表问题来自一份报告，需在结论中说明领域与模板局限。",
    ],
    "questions": [q.model_dump() for q in request.questions],
}
path = ROOT / "datasets/public_questions.json"
path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
print(path, dict(counts), "dev=20 test=80")
