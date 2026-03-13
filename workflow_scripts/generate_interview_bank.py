from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import yaml


QUESTION_PATTERN = re.compile(r"^#\s*题目\d+[：:]\s*(.+?)\s*$", re.MULTILINE)
EXCLUDED_REPORT_ROOTS = {"论文精读", "体系化课程", "00_行业洞察", "专题题库"}


@dataclass(frozen=True)
class QuestionItem:
    question: str
    module: str
    level: str
    source: str
    companies: tuple[str, ...]


MODULE_GUIDES = {
    "Transformer 结构与注意力": {
        "focus": "重点回答 attention 变体、归一化、激活函数、参数效率与训练/推理 trade-off。",
        "follow_ups": ["为什么 Decoder-only 成为主流？", "GQA、MQA、MHA 的差别是什么？"],
        "companies": ("字节跳动", "阿里巴巴", "腾讯", "百度"),
    },
    "推理优化与部署": {
        "focus": "重点回答 KV Cache、FlashAttention、量化、并发调度、显存瓶颈与吞吐-时延权衡。",
        "follow_ups": ["PagedAttention 解决了什么问题？", "如何设计线上首 token 延迟优化方案？"],
        "companies": ("字节跳动", "百度", "美团", "快手"),
    },
    "长上下文与位置编码": {
        "focus": "重点回答 RoPE/ALiBi、外推、插值、长文本训练策略与检索式补偿方案。",
        "follow_ups": ["为什么直接外推会崩？", "长上下文问题怎么评测？"],
        "companies": ("字节跳动", "阿里巴巴", "腾讯"),
    },
    "训练并行与系统": {
        "focus": "重点回答 DP/TP/PP/ZeRO、MoE 训练、checkpoint、混合精度与故障恢复。",
        "follow_ups": ["100B 模型怎么切并行？", "训练吞吐和稳定性如何兼顾？"],
        "companies": ("阿里巴巴", "百度", "字节跳动", "美团"),
    },
    "后训练与对齐": {
        "focus": "重点回答 SFT、DPO、PPO、奖励模型、拒答、安全与偏好数据构造。",
        "follow_ups": ["为什么 SFT 后还要 RLHF？", "DPO 和 PPO 各自适用什么场景？"],
        "companies": ("字节跳动", "阿里巴巴", "腾讯", "百度"),
    },
    "RAG 与 Agent": {
        "focus": "重点回答索引构建、召回排序、上下文拼接、工具调用、规划与记忆机制。",
        "follow_ups": ["RAG 如何做评测闭环？", "Agent 为什么容易失败在工具调用？"],
        "companies": ("字节跳动", "腾讯", "美团", "快手"),
    },
    "多模态与视觉语言": {
        "focus": "重点回答图文对齐、 projector、视觉编码器冻结策略、多图/视频理解与 OCR。",
        "follow_ups": ["VLM 的训练阶段通常怎么拆？", "多模态 hallucination 如何缓解？"],
        "companies": ("字节跳动", "阿里巴巴", "百度"),
    },
    "数据工程与评测": {
        "focus": "重点回答数据配比、去重、清洗、合成增强、离线/在线评测与安全红线。",
        "follow_ups": ["高质量数据怎么定义？", "线上评测指标和离线指标不一致怎么办？"],
        "companies": ("字节跳动", "腾讯", "美团"),
    },
    "生成模型与扩散": {
        "focus": "重点回答 diffusion 基本过程、噪声调度、约束生成、inpainting 与一致性问题。",
        "follow_ups": ["为什么中间步骤难控制？", "约束生成最容易在哪些地方失效？"],
        "companies": ("字节跳动", "快手", "百度"),
    },
    "经典机器学习与编程": {
        "focus": "重点回答基础算法、数据结构、损失函数、优化器和常见机器学习建模套路。",
        "follow_ups": ["如果要线上落地，你会怎么改？", "时间复杂度和空间复杂度怎么分析？"],
        "companies": ("字节跳动", "阿里巴巴", "腾讯", "美团"),
    },
}


KEYWORD_TO_MODULE = [
    ("代码题|leetcode|合并区间|三数之和|升序链表|等和子集|target的组合|topk|二叉树|动态规划|回溯|贪心", "经典机器学习与编程"),
    ("多轮对话|function call|tool call|tool use|agent|rag|graphrag|知识图谱|向量检索|召回|重排|记忆机制", "RAG 与 Agent"),
    ("灾难性遗忘|数据配比|样本多样性|数据集|数据增强|评测|benchmark|正确率|难样本|清洗", "数据工程与评测"),
    ("kv cache|gqa|mqa|flashattention|pagedattention|量化|量化感知|推理|吞吐|延迟|vllm|部署", "推理优化与部署"),
    ("rope|alibi|长上下文|位置编码|yarn|ntk|context window|lost in the middle", "长上下文与位置编码"),
    ("zero|deepspeed|数据并行|张量并行|流水线并行|moe|checkpoint|混合精度|训练优化", "训练并行与系统"),
    ("sft|rlhf|dpo|ppo|orpo|kto|奖励模型|对齐|偏好|拒答|安全", "后训练与对齐"),
    ("多模态|视觉|图文|视频|语音|vlm|ocr|q-former|projector", "多模态与视觉语言"),
    ("diffusion|扩散|inpainting|生成中间步骤|补全|重采样", "生成模型与扩散"),
    ("decoder-only|transformer|qwen|llama|lora|sft数据|swi?glu|rmsnorm|attention|位置编码", "Transformer 结构与注意力"),
    ("链表|排序|机器学习|xgboost|逻辑回归|svm", "经典机器学习与编程"),
]


DISTILLED_QUESTIONS = [
    ("讲一下 Transformer block 里 RMSNorm、SwiGLU、RoPE 分别解决了什么问题", "Transformer 结构与注意力", "高频"),
    ("为什么 GQA 会成为当前主流大模型的默认选择", "Transformer 结构与注意力", "高频"),
    ("MoE 模型相比 dense 模型的核心收益和训练难点是什么", "Transformer 结构与注意力", "中高频"),
    ("你怎么向业务同学解释 KV Cache 不是越大越好", "推理优化与部署", "高频"),
    ("线上服务里吞吐、首 token 延迟、显存占用三者如何权衡", "推理优化与部署", "高频"),
    ("如果让你做一个 7B 模型的 INT4 部署方案，你会怎么验证质量回退", "推理优化与部署", "中高频"),
    ("为什么 RoPE 扩到 128K 之后还经常要配合检索和摘要记忆", "长上下文与位置编码", "中高频"),
    ("ALiBi 和 RoPE 在工程上各自有什么优缺点", "长上下文与位置编码", "中高频"),
    ("Needle in a Haystack 指标能说明什么，不能说明什么", "长上下文与位置编码", "中高频"),
    ("如果给你 256 张卡训练 70B 模型，你会怎么搭 3D 并行", "训练并行与系统", "高频"),
    ("为什么 ZeRO-3 更省显存，但不一定是吞吐最优", "训练并行与系统", "高频"),
    ("训练超长上下文模型时，如何控制梯度爆炸和不稳定", "训练并行与系统", "中高频"),
    ("偏好数据很少时，你会优先做 DPO、PPO 还是拒绝采样", "后训练与对齐", "高频"),
    ("奖励模型最容易学偏什么，怎么发现奖励黑客行为", "后训练与对齐", "中高频"),
    ("怎样设计拒答数据，既安全又不把模型训成过度保守", "后训练与对齐", "中高频"),
    ("RAG 系统中召回、重排、生成三段各应该看什么指标", "RAG 与 Agent", "高频"),
    ("为什么很多 Agent 系统不是死在规划，而是死在工具接口", "RAG 与 Agent", "中高频"),
    ("多轮对话里记忆机制应该做在 prompt、RAG 还是模型内部", "RAG 与 Agent", "中高频"),
    ("一个 VLM 项目通常怎么拆成视觉预训练、对齐、指令微调三阶段", "多模态与视觉语言", "中高频"),
    ("多图输入和视频输入的难点分别是什么", "多模态与视觉语言", "中高频"),
    ("如果 OCR 错了，VLM 后续回答往往会怎样失真", "多模态与视觉语言", "中高频"),
    ("做 SFT 时通用数据、垂类数据和安全数据怎么配比更稳", "数据工程与评测", "高频"),
    ("为什么离线 benchmark 提升了，线上用户体验未必提升", "数据工程与评测", "高频"),
    ("合成数据增强什么时候最有效，什么时候反而会伤模型", "数据工程与评测", "中高频"),
    ("扩散模型做局部补全时，边界连贯性为什么特别难处理", "生成模型与扩散", "中高频"),
    ("如果已知起点和终点，为什么生成中间步骤往往不自然", "生成模型与扩散", "中高频"),
    ("你会如何把几何先验写进扩散模型的 loss 或约束里", "生成模型与扩散", "中高频"),
    ("逻辑回归、SVM、XGBoost 各自适合什么数据规模和特征形态", "经典机器学习与编程", "中高频"),
    ("TopK 问题、合并区间、K 路归并为什么在算法岗里反复出现", "经典机器学习与编程", "高频"),
    ("如果线上特征分布变了，你会先做模型回滚还是数据诊断", "经典机器学习与编程", "中高频"),
]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_question(text: str) -> str:
    lowered = re.sub(r"\s+", "", text).strip().lower()
    return re.sub(r"[，。！？,.?:：、“”‘’\"'`]", "", lowered)


def classify_module(text: str) -> str:
    normalized = text.lower()
    for pattern, module in KEYWORD_TO_MODULE:
        if re.search(pattern, normalized):
            return module
    return "Transformer 结构与注意力"


def iter_existing_questions(reports_dir: Path) -> Iterable[QuestionItem]:
    for md_path in reports_dir.rglob("*.md"):
        relative = md_path.relative_to(reports_dir)
        if not relative.parts:
            continue
        if relative.parts[0] in EXCLUDED_REPORT_ROOTS:
            continue

        company = relative.parts[0]
        content = md_path.read_text(encoding="utf-8")
        for matched in QUESTION_PATTERN.findall(content):
            question = matched.strip()
            if not question:
                continue
            yield QuestionItem(
                question=question,
                module=classify_module(question),
                level="高频",
                source=f"仓库沉淀：{relative.as_posix()}",
                companies=(company,),
            )


def build_distilled_questions() -> list[QuestionItem]:
    items: list[QuestionItem] = []
    for question, module, level in DISTILLED_QUESTIONS:
        guide = MODULE_GUIDES[module]
        items.append(
            QuestionItem(
                question=question,
                module=module,
                level=level,
                source="蒸馏补充",
                companies=guide["companies"],
            )
        )
    return items


def dedupe_questions(items: Iterable[QuestionItem]) -> list[QuestionItem]:
    merged: dict[str, QuestionItem] = {}
    for item in items:
        key = normalize_question(item.question)
        if key not in merged:
            merged[key] = item
            continue

        existing = merged[key]
        merged_companies = tuple(dict.fromkeys(existing.companies + item.companies))
        better_source = existing.source
        if existing.source == "蒸馏补充" and item.source != "蒸馏补充":
            better_source = item.source
        merged[key] = QuestionItem(
            question=existing.question,
            module=existing.module,
            level=existing.level if existing.level == "高频" else item.level,
            source=better_source,
            companies=merged_companies,
        )
    return sorted(merged.values(), key=lambda item: (item.module, item.question))


def build_stats(questions: list[QuestionItem]) -> dict[str, list[QuestionItem]]:
    grouped: dict[str, list[QuestionItem]] = defaultdict(list)
    for item in questions:
        grouped[item.module].append(item)
    return dict(sorted(grouped.items(), key=lambda pair: pair[0]))


def render_table_rows(questions: list[QuestionItem]) -> str:
    lines = [
        "| 序号 | 题目 | 模块 | 难度 | 常见公司 | 来源 |",
        "| ---: | --- | --- | --- | --- | --- |",
    ]
    for index, item in enumerate(questions, start=1):
        companies = " / ".join(item.companies[:4])
        lines.append(
            f"| {index} | {item.question} | {item.module} | {item.level} | {companies} | {item.source} |"
        )
    return "\n".join(lines)


def render_overview(questions: list[QuestionItem], module_map: dict[str, list[QuestionItem]]) -> str:
    stats_lines = [
        "| 模块 | 题目数 | 建议优先级 |",
        "| --- | ---: | --- |",
    ]
    for module, items in module_map.items():
        priority = "优先准备" if len(items) >= 4 else "按需准备"
        stats_lines.append(f"| {module} | {len(items)} | {priority} |")

    hot_questions = questions[: min(40, len(questions))]
    return f"""# 大厂 AI 算法高频题总表

> 自动生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
> 生成方式：扫描仓库现有面经题目 + 蒸馏补齐高频模块
> 使用建议：先刷“推理优化 / 对齐 / RAG / 训练并行”四类，再补“多模态 / 长上下文 / 数据工程”

## 本次升级带来的题库能力

- 不再只按单篇面经浏览，而是新增“按模块聚合”的高频题单。
- 自动复用仓库里已经整理好的题目，避免重复手工维护。
- 对仓库中较少覆盖的模块做蒸馏补齐，形成可持续迭代的统一题库。

## 模块分布

{chr(10).join(stats_lines)}

## 高频题速览

{render_table_rows(hot_questions)}
"""


def render_module_page(module_map: dict[str, list[QuestionItem]]) -> str:
    lines = [
        "# 大厂 AI 算法高频题按模块分类",
        "",
        f"> 自动生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "> 本页适合做定向复习：每个模块都给出考察点、典型追问和高频题清单。",
        "",
    ]

    for module, items in module_map.items():
        guide = MODULE_GUIDES[module]
        lines.append(f"## {module} ({len(items)})")
        lines.append("")
        lines.append(f"**面试官在看什么**：{guide['focus']}")
        lines.append("")
        lines.append(f"**典型追问**：{'；'.join(guide['follow_ups'])}")
        lines.append("")
        for index, item in enumerate(items, start=1):
            companies = " / ".join(item.companies[:4])
            lines.append(f"### {index}. {item.question}")
            lines.append("")
            lines.append(f"- 难度：{item.level}")
            lines.append(f"- 常见公司：{companies}")
            lines.append(f"- 来源：{item.source}")
            lines.append("")

    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    config = load_config(base_dir / "config.yaml")
    reports_root = base_dir / config.get("paths", {}).get("reports", "reports")

    all_questions = dedupe_questions(
        list(iter_existing_questions(reports_root)) + build_distilled_questions()
    )
    module_map = build_stats(all_questions)

    output_dir = reports_root / "专题题库" / "AI算法高频题"
    date_prefix = datetime.now().strftime("%Y%m%d")
    write_text(output_dir / f"{date_prefix}_大厂AI算法高频题总表.md", render_overview(all_questions, module_map))
    write_text(output_dir / f"{date_prefix}_大厂AI算法高频题按模块分类.md", render_module_page(module_map))

    print(f"[√] 已生成 {len(all_questions)} 道高频题，输出目录：{output_dir}")


if __name__ == "__main__":
    main()
