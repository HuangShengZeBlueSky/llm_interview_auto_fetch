from __future__ import annotations

import json
import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import yaml
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


REPO_SPECS = [
    (
        "DeepLearing-Interview-Awesome-2024",
        "https://github.com/315386775/DeepLearing-Interview-Awesome-2024.git",
    ),
    (
        "LLMs_interview_notes",
        "https://github.com/km1994/LLMs_interview_notes.git",
    ),
]


DEEPLEARNING_MODULES = [
    ("DeepLearing-Interview-Awesome-2024_LLMs专题", "LLMs/Reference.md"),
    ("DeepLearing-Interview-Awesome-2024_Agent专题", "LLMs/Agent.md"),
    ("DeepLearing-Interview-Awesome-2024_视觉感知专题", "VisionPerception/Reference.md"),
    ("DeepLearing-Interview-Awesome-2024_深度学习基础专题", "DeepLearning/Reference.md"),
    ("DeepLearing-Interview-Awesome-2024_行业算法专题", "IndustryAlgorithm/Reference.md"),
    ("DeepLearing-Interview-Awesome-2024_手撕代码专题", "CodeAnything/Reference.md"),
    ("DeepLearing-Interview-Awesome-2024_开源项目专题", "AwesomeProjects/Reference.md"),
]


QUESTION_LIKE_KEYWORDS = (
    "什么",
    "为何",
    "为什么",
    "如何",
    "区别",
    "优点",
    "缺点",
    "作用",
    "原理",
    "介绍",
    "简述",
    "怎么",
    "能否",
    "是否",
    "有哪些",
    "场景",
    "流程",
    "实现",
    "公式",
    "训练",
    "推理",
)


LOCAL_REPORT_QUESTION_RE = re.compile(r"^#\s*题目\d+[：:]\s*(.+?)\s*$", re.MULTILINE)
TOP_HEADING_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
QUESTION_INDEX_LINE_RE = re.compile(r"\[\*\*(.+?)\*\*\]\((.+?)\)")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
RAW_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>\n]*>")
LOCAL_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((?!https?://)([^)]+)\)")


@dataclass
class QAItem:
    question: str
    answer: str
    source_label: str
    source_url: str


@dataclass
class NotesSection:
    major_group: str
    title: str
    source_url: str
    questions: list[str]


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_name(text: str) -> str:
    cleaned = re.sub(r'[\\/*?:"<>|]+', "", text).strip()
    return cleaned or "untitled"


def clean_markdown_text(text: str) -> str:
    text = MARKDOWN_LINK_RE.sub(lambda m: m.group(1), text)
    text = text.replace("**", "").replace("`", "")
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    text = clean_markdown_text(text).lower()
    text = re.sub(r"[（）()【】\[\]《》<>“”‘’\"'`·\-—_,.;:：，。！？!?/\\|]", "", text)
    text = re.sub(r"\s+", "", text)
    return text


def is_question_like(text: str) -> bool:
    candidate = clean_markdown_text(text)
    if len(candidate) < 4:
        return False
    if candidate.startswith("点击查看答案"):
        return False
    if "http" in candidate.lower():
        return False
    if candidate.endswith(("？", "?")):
        return True
    return any(keyword in candidate for keyword in QUESTION_LIKE_KEYWORDS)


def run_git_command(args: list[str], cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=str(cwd) if cwd else None, check=True, capture_output=True, text=True, encoding="utf-8")


def ensure_repo_cloned(base_dir: Path, repo_name: str, repo_url: str) -> Path:
    repo_root = base_dir / "external" / repo_name
    repo_root.parent.mkdir(parents=True, exist_ok=True)
    if repo_root.exists():
        run_git_command(["git", "-C", str(repo_root), "pull", "--ff-only"])
    else:
        run_git_command(["git", "clone", "--depth", "1", repo_url, str(repo_root)])
    return repo_root


def parse_heading_blocks(markdown_text: str) -> list[tuple[str, str]]:
    matches = list(TOP_HEADING_RE.finditer(markdown_text))
    blocks = []
    for index, matched in enumerate(matches):
        title = clean_markdown_text(matched.group(1))
        start = matched.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown_text)
        body = markdown_text[start:end].strip()
        blocks.append((title, body))
    return blocks


def parse_repo1_qas(repo_root: Path) -> list[QAItem]:
    results: list[QAItem] = []
    for module_name, relative_path in DEEPLEARNING_MODULES:
        target_path = repo_root / relative_path
        if not target_path.exists():
            continue
        content = target_path.read_text(encoding="utf-8")
        for title, body in parse_heading_blocks(content):
            question = re.sub(r"^\d+\.\s*", "", title).strip()
            results.append(
                QAItem(
                    question=question,
                    answer=body.strip(),
                    source_label=f"{module_name}:{relative_path}",
                    source_url=f"https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/{relative_path.replace(os.sep, '/')}",
                )
            )
    return results


def parse_local_reports_qas(base_dir: Path) -> list[QAItem]:
    reports_dir = base_dir / "reports"
    if not reports_dir.exists():
        return []

    items: list[QAItem] = []
    for md_path in reports_dir.rglob("*.md"):
        relative = md_path.relative_to(reports_dir)
        if relative.parts and relative.parts[0] == "论文精读":
            continue
        content = md_path.read_text(encoding="utf-8")
        matches = list(LOCAL_REPORT_QUESTION_RE.finditer(content))
        if not matches:
            continue
        for index, matched in enumerate(matches):
            question = clean_markdown_text(matched.group(1))
            start = matched.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            answer = content[start:end].strip()
            items.append(
                QAItem(
                    question=question,
                    answer=answer,
                    source_label=f"本仓库沉淀:{relative.as_posix()}",
                    source_url="",
                )
            )
    return items


def similarity_score(left: str, right: str) -> float:
    norm_left = normalize_text(left)
    norm_right = normalize_text(right)
    if not norm_left or not norm_right:
        return 0.0
    if norm_left in norm_right or norm_right in norm_left:
        shorter = min(len(norm_left), len(norm_right))
        longer = max(len(norm_left), len(norm_right))
        return 0.72 + 0.25 * (shorter / max(longer, 1))
    return SequenceMatcher(None, norm_left, norm_right).ratio()


def best_answer_match(question: str, answer_bank: list[QAItem], threshold: float = 0.60) -> tuple[QAItem | None, float]:
    best_item = None
    best_score = 0.0
    for item in answer_bank:
        score = similarity_score(question, item.question)
        if score > best_score:
            best_score = score
            best_item = item
    if best_score >= threshold:
        return best_item, best_score
    return None, best_score


def trim_answer(text: str, limit: int = 1200) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def sanitize_markdown_html(text: str) -> str:
    sanitized = RAW_HTML_TAG_RE.sub(
        lambda matched: matched.group(0).replace("<", "&lt;").replace(">", "&gt;"),
        text,
    )
    return LOCAL_IMAGE_RE.sub(lambda matched: f"> 图示见原仓库资源：{matched.group(2)}", sanitized)


def init_client(config: dict) -> tuple[OpenAI | None, str]:
    llm_config = config.get("llm", {})
    api_key_env_name = llm_config.get("api_key_env_var", "LLM_API_KEY")
    api_key = os.environ.get(api_key_env_name) or llm_config.get("api_key")
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        return None, llm_config.get("model", "")
    client = OpenAI(
        api_key=api_key,
        base_url=llm_config.get("base_url"),
        timeout=300.0,
    )
    return client, llm_config.get("model", "")


def parse_json_payload(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        matched = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
        if matched:
            return json.loads(matched.group(1).strip())
    raise ValueError("无法从模型输出中解析 JSON")


def parse_structured_answers(text: str) -> list[tuple[str, str]]:
    pattern = re.compile(
        r"^###\s*\d+.*?\nQ:\s*(.*?)\nA:\s*(.*?)(?=^###\s*\d+|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    pairs = []
    for matched in pattern.finditer(text):
        question = clean_markdown_text(matched.group(1))
        answer = matched.group(2).strip()
        if question and answer:
            pairs.append((question, answer))
    return pairs


def generate_answers_with_llm(
    client: OpenAI | None,
    model_name: str,
    major_group: str,
    section_title: str,
    questions: list[str],
) -> dict[str, str]:
    if client is None or not questions:
        return {question: "未匹配到现成答案，且当前未配置可用 LLM，建议后续补充解析。" for question in questions}

    numbered_questions = "\n".join(f"{index}. {question}" for index, question in enumerate(questions, start=1))
    system_prompt = (
        "你是资深 AI 算法面试官与大模型工程师。"
        "请针对给定的一组中文面试题，给出简洁但靠谱的中文解析。"
        "严格按如下文本格式输出，不要使用 JSON，不要省略任何题目：\n"
        "### 1\nQ: 问题原文\nA: 对应答案\n"
        "### 2\nQ: 问题原文\nA: 对应答案\n"
        "answer 要满足：1）120-220字；2）先给结论，再给关键点；3）适合大厂 AI 算法面试；4）避免空话。"
    )
    user_prompt = (
        f"专题大类：{major_group}\n"
        f"专题小节：{section_title}\n"
        f"请回答下面这些问题：\n{numbered_questions}"
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        parsed_pairs = parse_structured_answers(response.choices[0].message.content)
        result = {question: answer for question, answer in parsed_pairs}
        if len(result) >= len(questions):
            return {question: result.get(clean_markdown_text(question), result.get(question, "未成功生成自动解析，建议后续补充。")) for question in questions}
    except Exception:
        result = {}

    if len(questions) == 1:
        return {questions[0]: "未成功生成自动解析，建议后续补充。"}

    merged: dict[str, str] = {}
    for batch in chunk_list(questions, 1):
        merged.update(generate_answers_with_llm(client, model_name, major_group, section_title, batch))
    return merged


def generate_major_group_summary(
    client: OpenAI | None,
    model_name: str,
    major_group: str,
    sections: list[NotesSection],
) -> str:
    representative_questions: list[str] = []
    for section in sections:
        representative_questions.extend(section.questions[:2])
        if len(representative_questions) >= 10:
            break
    representative_questions = representative_questions[:10]

    if client is None or not representative_questions:
        return "当前未配置可用 LLM，已保留原始题目清单与可匹配的现成答案，后续可继续补全专题总结。"

    numbered_questions = "\n".join(f"{index}. {question}" for index, question in enumerate(representative_questions, start=1))
    system_prompt = (
        "你是资深 AI 算法面试官。"
        "请根据一个大专题下的代表性问题，生成一段适合知识库网页阅读的中文总结。"
        "输出必须是 Markdown，结构固定为："
        "\n## 本专题考什么\n- ..."
        "\n## 答题主线\n- ..."
        "\n## 代表题答法\n### 问题1\n..."
        "\n### 问题2\n..."
        "\n只总结最值得准备的内容，避免空话。"
    )
    user_prompt = (
        f"专题名称：{major_group}\n"
        "请根据这些代表题，输出本专题的核心知识点、答题主线和代表题答法：\n"
        f"{numbered_questions}"
    )
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "自动专题总结生成失败，本页仍保留全量原始题目清单与可复用答案。"


def parse_repo2_sections(repo_root: Path) -> list[NotesSection]:
    readme_path = repo_root / "README.md"
    lines = readme_path.read_text(encoding="utf-8").splitlines()
    sections: list[NotesSection] = []
    current_major = "LLMs_interview_notes"
    current_section: NotesSection | None = None

    def flush_current() -> None:
        nonlocal current_section
        if current_section and current_section.questions:
            current_section.questions = list(dict.fromkeys(current_section.questions))
            sections.append(current_section)
        current_section = None

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip():
            continue

        if line.startswith("## "):
            flush_current()
            current_major = clean_markdown_text(line[3:])
            continue

        heading_match = re.match(r"^(###|####)\s+(.+)$", line)
        if heading_match:
            flush_current()
            heading_body = clean_markdown_text(heading_match.group(2))
            link_match = MARKDOWN_LINK_RE.search(heading_match.group(2))
            current_section = NotesSection(
                major_group=current_major,
                title=heading_body,
                source_url=link_match.group(2) if link_match else "",
                questions=[],
            )
            continue

        if current_section is None:
            continue

        if "点击查看答案" in line:
            continue

        bullet_match = re.match(r"^\s*-\s+(.+?)\s*$", line)
        if not bullet_match:
            continue
        question = clean_markdown_text(bullet_match.group(1))
        if is_question_like(question):
            current_section.questions.append(question)

    flush_current()
    return sections


def chunk_list(items: list[str], chunk_size: int) -> list[list[str]]:
    return [items[index:index + chunk_size] for index in range(0, len(items), chunk_size)]


def build_repo2_answer_pages(
    sections: list[NotesSection],
    answer_bank: list[QAItem],
    client: OpenAI | None,
    model_name: str,
) -> list[tuple[str, str]]:
    pages: list[tuple[str, str]] = []
    by_major: dict[str, list[NotesSection]] = defaultdict(list)
    for section in sections:
        by_major[section.major_group].append(section)

    for major_group, major_sections in by_major.items():
        lines = [
            f"# LLMs_interview_notes 提取：{major_group}",
            "",
            "> 来源仓库：https://github.com/km1994/LLMs_interview_notes",
            "> 说明：该仓库大量内容以“题目目录 + 外链答案”形式存在，本页保留全量题目清单，并优先复用本仓库现成答案，再补充专题级总结。",
            "",
            "## 专题总结",
            "",
            generate_major_group_summary(client, model_name, major_group, major_sections),
            "",
        ]
        for section in major_sections:
            lines.append(f"## {section.title}")
            lines.append("")
            if section.source_url:
                lines.append(f"- 来源链接：{section.source_url}")
            lines.append(f"- 题目数：{len(section.questions)}")
            lines.append("")
            lines.append("### 原始题目")
            lines.append("")
            for index, question in enumerate(section.questions, start=1):
                lines.append(f"{index}. {question}")
            lines.append("")

            matched_examples: list[tuple[str, str, str, float]] = []
            for question in section.questions:
                matched_item, score = best_answer_match(question, answer_bank, threshold=0.68)
                if matched_item is not None:
                    matched_examples.append(
                        (
                            question,
                            trim_answer(matched_item.answer, 700),
                            matched_item.source_label,
                            score,
                        )
                    )
                if len(matched_examples) >= 2:
                    break

            if matched_examples:
                lines.append("### 可直接复用的答案")
                lines.append("")
                for index, (question, answer_text, answer_source, score) in enumerate(matched_examples, start=1):
                    lines.append(f"#### 示例 {index}. {question}")
                    lines.append("")
                    lines.append(answer_text)
                    lines.append("")
                    lines.append(f"> 匹配来源：{answer_source} | 匹配分数：{score:.2f}")
                    lines.append("")

        pages.append((major_group, "\n".join(lines)))
    return pages


def render_repo1_page(title: str, qas: list[QAItem], source_url: str) -> str:
    lines = [
        f"# GitHub 提取：{title}",
        "",
        f"> 来源仓库：[DeepLearing-Interview-Awesome-2024]({source_url})",
        f"> 本页共整理 {len(qas)} 道题，尽量保留原仓库答案；若原仓库缺少解析，则自动补齐。",
        "",
    ]
    for index, item in enumerate(qas, start=1):
        lines.append(f"## {index}. {item.question}")
        lines.append("")
        if item.answer:
            lines.append(item.answer.strip())
        else:
            lines.append("原仓库未提供解析，本次未匹配到可用答案。")
        lines.append("")
    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = sanitize_markdown_html(content.strip()) + "\n"
    path.write_text(sanitized, encoding="utf-8", newline="\n")


def render_overview(repo1_pages: list[tuple[str, int]], repo2_pages: list[tuple[str, int]]) -> str:
    lines = [
        "# GitHub 面经仓库提取总览",
        "",
        f"> 自动生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "> 提取来源：",
        "> - https://github.com/315386775/DeepLearing-Interview-Awesome-2024",
        "> - https://github.com/km1994/LLMs_interview_notes",
        "",
        "## 本次导入策略",
        "",
        "- 对于仓库内已有答案的题目，直接抽取并归一化成站点资料。",
        "- 对于只有题目目录的内容，优先匹配本仓库已有题解与另一个仓库中的近似答案。",
        "- 对于仍然缺失的题目，再用大模型自动补全简洁解析。",
        "",
        "## 产出页概览",
        "",
        "### DeepLearing-Interview-Awesome-2024",
        "",
    ]
    for title, count in repo1_pages:
        lines.append(f"- {title} | {count} 道题")
    lines.extend(["", "### LLMs_interview_notes", ""])
    for title, count in repo2_pages:
        lines.append(f"- {title} | {count} 道题")
    return "\n".join(lines)


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    config = load_config(base_dir / "config.yaml")
    client, model_name = init_client(config)

    repo_paths = {
        repo_name: ensure_repo_cloned(base_dir, repo_name, repo_url)
        for repo_name, repo_url in REPO_SPECS
    }

    repo1_root = repo_paths["DeepLearing-Interview-Awesome-2024"]
    repo2_root = repo_paths["LLMs_interview_notes"]

    repo1_qas = parse_repo1_qas(repo1_root)
    local_qas = parse_local_reports_qas(base_dir)
    answer_bank = repo1_qas + local_qas

    output_dir = base_dir / "reports" / "专题题库" / "GitHub面经仓"
    date_prefix = datetime.now().strftime("%Y%m%d")

    repo1_page_meta: list[tuple[str, int]] = []
    for module_title, relative_path in DEEPLEARNING_MODULES:
        source_qas = [item for item in repo1_qas if relative_path.replace("\\", "/") in item.source_label]
        if not source_qas:
            continue

        missing_questions = [item.question for item in source_qas if not item.answer.strip()]
        generated_answers: dict[str, str] = {}
        for batch in chunk_list(missing_questions, 4):
            generated_answers.update(
                generate_answers_with_llm(client, model_name, "DeepLearing-Interview-Awesome-2024", module_title, batch)
            )

        final_qas: list[QAItem] = []
        for item in source_qas:
            answer = item.answer.strip()
            if not answer:
                answer = generated_answers.get(item.question, "未生成解析。")
            final_qas.append(QAItem(item.question, answer, item.source_label, item.source_url))

        output_path = output_dir / f"{date_prefix}_{safe_name(module_title)}.md"
        source_url = f"https://github.com/315386775/DeepLearing-Interview-Awesome-2024/blob/main/{relative_path.replace(os.sep, '/')}"
        write_text(output_path, render_repo1_page(module_title, final_qas, source_url))
        repo1_page_meta.append((module_title, len(final_qas)))

    repo2_sections = parse_repo2_sections(repo2_root)
    repo2_pages = build_repo2_answer_pages(repo2_sections, answer_bank, client, model_name)
    repo2_page_meta: list[tuple[str, int]] = []
    for major_group, page_content in repo2_pages:
        question_count = sum(section.questions.__len__() for section in repo2_sections if section.major_group == major_group)
        output_path = output_dir / f"{date_prefix}_{safe_name('LLMs_interview_notes_' + major_group)}.md"
        write_text(output_path, page_content)
        repo2_page_meta.append((major_group, question_count))

    overview_path = output_dir / f"{date_prefix}_GitHub面经仓库提取总览.md"
    write_text(overview_path, render_overview(repo1_page_meta, repo2_page_meta))

    print(f"[√] 已完成两个 GitHub 仓库的提取与汇总，输出目录：{output_dir}")


if __name__ == "__main__":
    main()
