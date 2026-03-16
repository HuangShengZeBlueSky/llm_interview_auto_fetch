from __future__ import annotations

import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from hashlib import md5
from html import unescape
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(".env")


REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/134.0.0.0 Safari/537.36"
    )
}

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

TOP_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
LOCAL_REPORT_QUESTION_RE = re.compile(r"^#\s*题目\d+[：:]\s*(.+?)\s*$", re.MULTILINE)
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
RAW_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>\n]*>")
LOCAL_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((?!https?://)([^)]+)\)")
QUESTION_LINE_RE = re.compile(r"^\s*(?:[-*]|(?:\d+|[一二三四五六七八九十]+)[\.、:：\)]?)\s*(.+?)\s*$")

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
    "微调",
    "并行",
    "部署",
    "损失",
    "注意力",
    "蒸馏",
    "幻觉",
    "量化",
    "RAG",
    "Agent",
    "Tokenizer",
    "MoE",
    "RLHF",
)

TAG_KEYWORDS = [
    ("RAG与向量检索", ("rag", "检索", "向量库", "embedding", "召回", "rerank")),
    ("Agent与工具调用", ("agent", "工具调用", "function calling", "langchain", "多轮对话", "tool")),
    ("模型微调", ("sft", "微调", "lora", "qlora", "ptuning", "prompt tuning", "adapter", "peft")),
    ("Transformer结构", ("transformer", "attention", "位置编码", "rope", "layernorm", "flashattention")),
    ("推理优化与部署", ("推理", "部署", "kv cache", "vllm", "tensorrt", "量化", "paged attention")),
    ("训练并行与系统", ("分布式", "deepspeed", "fsdp", "并行", "显存", "oom", "compile", "cuda")),
    ("后训练与对齐", ("dpo", "ppo", "grpo", "reward model", "对齐", "偏好", "rlhf")),
    ("强化学习与RLHF", ("强化学习", "rlhf", "奖励模型", "policy", "价值函数")),
    ("多模态与视觉语言", ("多模态", "clip", "vlm", "视觉", "图像", "qformer")),
    ("生成模型与扩散", ("diffusion", "stable diffusion", "扩散", "unet", "vae")),
    ("数据工程与评测", ("数据集", "评测", "benchmark", "清洗", "标注", "采样")),
    ("经典算法与编程", ("手撕", "代码实现", "复杂度", "排序", "链表", "树", "dp")),
    ("LLM基础", ("llm", "大模型", "decoder only", "涌现", "token", "预训练", "chatgpt")),
    ("多模态基础", ("cross attention", "视觉编码器", "图文", "音频", "视频")),
    ("传统机器学习", ("svm", "xgboost", "逻辑回归", "adaboost", "随机森林")),
]

CHROME_PATHS = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]


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


@dataclass
class RawQuestion:
    question: str
    seed_answer: str
    page_group: str
    section_title: str
    source_name: str
    source_url: str
    module: str
    platform: str
    company: str
    seed_origin: str = ""


@dataclass
class SourceRef:
    source_name: str
    source_url: str
    page_group: str
    section_title: str
    platform: str
    company: str
    module: str


@dataclass
class QuestionAggregate:
    key: str
    question: str
    seed_answer: str
    seed_origin: str
    primary_tag: str
    module: str
    platform: str
    sources: list[SourceRef] = field(default_factory=list)


@dataclass
class QuestionCard:
    basics: list[str]
    detailed_answer: str
    case_simulation: str
    generated_from: str


@dataclass
class ExternalFetchResult:
    platform: str
    title: str
    company: str
    url: str
    status: str
    message: str
    question_count: int = 0


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def safe_name(text: str) -> str:
    cleaned = re.sub(r'[\\/*?:"<>|]+', "", text).strip()
    return cleaned or "untitled"


def clean_markdown_text(text: str) -> str:
    text = MARKDOWN_LINK_RE.sub(lambda m: m.group(1), text)
    text = text.replace("**", "").replace("`", "")
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", unescape(text)).strip()


def strip_html(html_text: str) -> str:
    text = html_text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = re.sub(r"</?(p|div|li|ul|ol|details|summary|blockquote|h\d)[^>]*>", "\n", text)
    text = re.sub(r"</?pre[^>]*>", "\n```text\n", text)
    text = re.sub(r"</?code[^>]*>", "`", text)
    text = re.sub(r"<a [^>]*>(.*?)</a>", r"\1", text, flags=re.S)
    text = re.sub(r"<img [^>]*>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = unescape(text)
    text = text.replace("\r", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_text(text: str) -> str:
    text = clean_markdown_text(text).lower()
    text = re.sub(r"[（）()【】\[\]{}<>“”‘’'\"`·,.;:：，。！？!？/\\|@#%^&*_+=\-~\s]", "", text)
    return text


def trim_text(text: str, limit: int = 1200) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def sanitize_markdown_html(text: str) -> str:
    sanitized = RAW_HTML_TAG_RE.sub(
        lambda matched: matched.group(0).replace("<", "&lt;").replace(">", "&gt;"),
        text,
    )
    return LOCAL_IMAGE_RE.sub(lambda matched: f"> 图示见原仓库资源：{matched.group(2)}", sanitized)


def is_question_like(text: str) -> bool:
    candidate = clean_markdown_text(text)
    if len(candidate) < 4:
        return False
    if candidate in {"前言", "...", "略"}:
        return False
    if candidate.endswith(("?", "？")):
        return True
    return any(keyword.lower() in candidate.lower() for keyword in QUESTION_LIKE_KEYWORDS)


def similarity_score(left: str, right: str) -> float:
    norm_left = normalize_text(left)
    norm_right = normalize_text(right)
    if not norm_left or not norm_right:
        return 0.0
    if norm_left == norm_right:
        return 1.0
    if norm_left in norm_right or norm_right in norm_left:
        shorter = min(len(norm_left), len(norm_right))
        longer = max(len(norm_left), len(norm_right))
        return 0.78 + 0.20 * (shorter / max(longer, 1))
    return SequenceMatcher(None, norm_left, norm_right).ratio()


def best_answer_match(question: str, answer_bank: list[QAItem], threshold: float = 0.66) -> tuple[QAItem | None, float]:
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


def chunk_list(items: list, chunk_size: int) -> list[list]:
    return [items[index:index + chunk_size] for index in range(0, len(items), chunk_size)]


def parse_json_payload(text: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        matched = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
        if matched:
            return json.loads(matched.group(1).strip())
    raise ValueError("无法从模型输出中解析 JSON")


def load_card_cache(cache_path: Path) -> dict[str, dict]:
    if not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text(encoding="utf-8"))


def save_card_cache(cache_path: Path, cache: dict[str, dict]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def run_git_command(args: list[str], cwd: Path | None = None) -> None:
    subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def ensure_repo_cloned(base_dir: Path, repo_name: str, repo_url: str) -> Path:
    repo_root = base_dir / "external" / repo_name
    repo_root.parent.mkdir(parents=True, exist_ok=True)
    if repo_root.exists():
        try:
            run_git_command(["git", "-C", str(repo_root), "pull", "--ff-only"])
        except subprocess.CalledProcessError:
            pass
    else:
        run_git_command(["git", "clone", "--depth", "1", repo_url, str(repo_root)])
    return repo_root


def parse_heading_blocks(markdown_text: str) -> list[tuple[str, str]]:
    matches = list(TOP_HEADING_RE.finditer(markdown_text))
    blocks: list[tuple[str, str]] = []
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
            question = re.sub(r"^\d+[\.、]\s*", "", title).strip()
            if len(question) < 3:
                continue
            results.append(
                QAItem(
                    question=question,
                    answer=body.strip(),
                    source_label=f"{module_name}:{relative_path}",
                    source_url=(
                        "https://github.com/315386775/DeepLearing-Interview-Awesome-2024/"
                        f"blob/main/{relative_path.replace(os.sep, '/')}"
                    ),
                )
            )
    return results


def parse_repo1_raw_questions(repo1_qas: list[QAItem]) -> list[RawQuestion]:
    raw_questions: list[RawQuestion] = []
    path_to_module = {
        relative_path.replace(os.sep, "/"): module_name
        for module_name, relative_path in DEEPLEARNING_MODULES
    }
    for item in repo1_qas:
        matched_module = next(
            (module_name for path_key, module_name in path_to_module.items() if path_key in item.source_label),
            "DeepLearing-Interview-Awesome-2024",
        )
        raw_questions.append(
            RawQuestion(
                question=item.question,
                seed_answer=item.answer.strip(),
                page_group=matched_module,
                section_title="原仓库题解",
                source_name="DeepLearing-Interview-Awesome-2024",
                source_url=item.source_url,
                module=matched_module,
                platform="github",
                company="未知",
                seed_origin="原仓库答案",
            )
        )
    return raw_questions


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

        if current_section is None or "点击查看答案" in line:
            continue

        bullet_match = re.match(r"^\s*-\s+(.+?)\s*$", line)
        if not bullet_match:
            continue

        question = clean_markdown_text(bullet_match.group(1))
        if question and question not in {"...", "……"} and is_question_like(question):
            current_section.questions.append(question)

    flush_current()
    return sections


def parse_repo2_raw_questions(sections: list[NotesSection], answer_bank: list[QAItem]) -> list[RawQuestion]:
    raw_questions: list[RawQuestion] = []
    for section in sections:
        for question in section.questions:
            matched_item, _ = best_answer_match(question, answer_bank, threshold=0.72)
            seed_answer = matched_item.answer if matched_item else ""
            seed_origin = f"近似题匹配：{matched_item.source_label}" if matched_item else ""
            raw_questions.append(
                RawQuestion(
                    question=question,
                    seed_answer=seed_answer,
                    page_group=section.major_group,
                    section_title=section.title,
                    source_name="LLMs_interview_notes",
                    source_url=section.source_url,
                    module=section.major_group,
                    platform="github",
                    company="未知",
                    seed_origin=seed_origin,
                )
            )
    return raw_questions


def find_browser_path() -> Path | None:
    for path in CHROME_PATHS:
        if path.exists():
            return path
    return None


def extract_json_object_from_html(html_text: str, marker: str) -> dict | None:
    marker_index = html_text.find(marker)
    if marker_index < 0:
        return None
    start = marker_index + len(marker)
    sub = html_text[start:]
    level = 0
    in_string = False
    escape = False
    started = False
    end = None
    for index, char in enumerate(sub):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            level += 1
            started = True
        elif char == "}":
            level -= 1
            if started and level == 0:
                end = index + 1
                break
    if end is None:
        return None
    payload = sub[:end]
    payload = re.sub(r"(?<=[:\[,])undefined(?=[,}\]])", "null", payload)
    payload = re.sub(r"(?<=[:\[,])NaN(?=[,}\]])", "null", payload)
    return json.loads(payload)


def fetch_nowcoder_source(url: str) -> tuple[str, str]:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    data = extract_json_object_from_html(response.text, "window.__INITIAL_STATE__=")
    if not data:
        raise ValueError("页面中未找到牛客内容状态")
    content_data = data["prefetchData"]["2"]["ssrCommonData"]["contentData"]
    show_message = content_data.get("showMessage", {})
    if show_message and show_message.get("message") not in {"", None}:
        raise ValueError(show_message["message"])
    title = clean_markdown_text(content_data.get("title", "牛客面经"))
    content = strip_html(content_data.get("content", ""))
    if not content:
        raise ValueError("牛客页面未提取到正文")
    return title, content


def fetch_zhihu_source(url: str) -> tuple[str, str]:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=30, allow_redirects=True)
    if response.status_code == 200 and "error" not in response.text[:400].lower():
        title_match = re.search(r"<title>(.*?)</title>", response.text, re.S)
        title = clean_markdown_text(title_match.group(1)) if title_match else "知乎面经"
        article_match = re.search(r"<article[^>]*>(.*?)</article>", response.text, re.S)
        if article_match:
            content = strip_html(article_match.group(1))
            if content:
                return title, content

    browser_path = find_browser_path()
    if browser_path is None:
        raise ValueError("本机未找到可用浏览器，知乎页面又直接访问受限")

    command = [
        str(browser_path),
        "--headless=new",
        "--disable-gpu",
        "--dump-dom",
        url,
    ]
    dump = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8").stdout
    if "请求存在异常" in dump or '"code":40362' in dump:
        raise ValueError("知乎触发反爬限制")
    title_match = re.search(r"<title>(.*?)</title>", dump, re.S)
    title = clean_markdown_text(title_match.group(1)) if title_match else "知乎面经"
    article_match = re.search(r"<article[^>]*>(.*?)</article>", dump, re.S)
    if not article_match:
        raise ValueError("知乎页面未提取到 article 正文")
    content = strip_html(article_match.group(1))
    if not content:
        raise ValueError("知乎正文为空")
    return title, content


def fetch_xiaohongshu_source(url: str) -> tuple[str, str]:
    browser_path = find_browser_path()
    if browser_path is None:
        raise ValueError("本机未找到可用浏览器，无法尝试抓取小红书页面")
    command = [
        str(browser_path),
        "--headless=new",
        "--disable-gpu",
        "--dump-dom",
        url,
    ]
    dump = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8").stdout
    if "安全限制" in dump or "IP存在风险" in dump:
        raise ValueError("小红书触发安全限制")
    title_match = re.search(r"<title>(.*?)</title>", dump, re.S)
    title = clean_markdown_text(title_match.group(1)) if title_match else "小红书面经"
    state = extract_json_object_from_html(dump, "window.__INITIAL_STATE__=")
    if state:
        note_state = state.get("note", {})
        note_map = note_state.get("noteDetailMap", {})
        for value in note_map.values():
            note = value.get("note") or {}
            content = note.get("desc", "") or note.get("content", "")
            if content:
                return title, strip_html(content)
    raise ValueError("小红书页面未提取到正文")


def load_external_sources(config_path: Path) -> list[dict]:
    if not config_path.exists():
        return []
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return payload.get("sources", [])


def extract_questions_from_article_text(text: str) -> list[str]:
    quoted_questions = re.findall(r"[“\"]([^”\"]+[？?])", text)
    questions = [clean_markdown_text(item) for item in quoted_questions if clean_markdown_text(item)]

    skip_fragments = (
        "获取更多面经",
        "点击查看答案",
        "深度解析",
        "考察逻辑",
        "核心洞察",
        "数据表明",
        "显著增加",
        "通常从",
        "这里的深层洞察",
        "公式为",
        "复数域的旋转诠释",
    )
    lines = [clean_markdown_text(line) for line in text.splitlines()]
    for line in lines:
        if not line or any(fragment in line for fragment in skip_fragments):
            continue
        matched = QUESTION_LINE_RE.match(line)
        candidate = matched.group(1).strip() if matched else line.strip()
        candidate = re.sub(r"^[\(（[]?\d+[\)）\].、:：-]?\s*", "", candidate)
        if len(candidate) < 4 or len(candidate) > 80:
            continue
        if candidate in {"前言", "背景", "总结", "反问", "论文拷打"}:
            continue
        if candidate.endswith("次"):
            continue
        if "。" in candidate and "？" not in candidate and "?" not in candidate:
            continue
        if "：" in candidate and not is_question_like(candidate):
            continue
        if is_question_like(candidate):
            questions.append(candidate)
    return list(dict.fromkeys(questions))


def fetch_external_sources(source_specs: list[dict]) -> tuple[list[RawQuestion], list[ExternalFetchResult]]:
    raw_questions: list[RawQuestion] = []
    results: list[ExternalFetchResult] = []

    for spec in source_specs:
        if spec.get("enabled", True) is False:
            continue
        platform = spec.get("platform", "").strip().lower()
        url = spec.get("url", "").strip()
        company = spec.get("company", "未知").strip() or "未知"
        title_hint = spec.get("title", "").strip()
        module = spec.get("module", "真实面经")
        if not platform or not url:
            continue

        try:
            if platform == "nowcoder":
                title, content = fetch_nowcoder_source(url)
            elif platform == "zhihu":
                title, content = fetch_zhihu_source(url)
            elif platform == "xiaohongshu":
                title, content = fetch_xiaohongshu_source(url)
            else:
                raise ValueError(f"暂不支持的平台：{platform}")

            page_title = title_hint or title
            questions = extract_questions_from_article_text(content)
            for question in questions:
                raw_questions.append(
                    RawQuestion(
                        question=question,
                        seed_answer="",
                        page_group=f"{platform}_{company}_{page_title}",
                        section_title=page_title,
                        source_name=f"真实面经_{platform}",
                        source_url=url,
                        module=module,
                        platform=platform,
                        company=company,
                        seed_origin="",
                    )
                )
            results.append(
                ExternalFetchResult(
                    platform=platform,
                    title=page_title,
                    company=company,
                    url=url,
                    status="success",
                    message="抓取成功",
                    question_count=len(questions),
                )
            )
        except Exception as exc:
            results.append(
                ExternalFetchResult(
                    platform=platform,
                    title=title_hint or "未命名来源",
                    company=company,
                    url=url,
                    status="failed",
                    message=str(exc),
                    question_count=0,
                )
            )

    return raw_questions, results


def infer_tag(*values: str) -> str:
    haystack = " ".join(value for value in values if value).lower()
    for tag, keywords in TAG_KEYWORDS:
        if any(keyword.lower() in haystack for keyword in keywords):
            return tag
    return "其他"


def choose_better_seed(existing_answer: str, candidate_answer: str) -> bool:
    return len(candidate_answer.strip()) > len(existing_answer.strip())


def merge_questions(raw_questions: list[RawQuestion], answer_bank: list[QAItem]) -> list[QuestionAggregate]:
    aggregates: list[QuestionAggregate] = []

    for raw in raw_questions:
        seed_answer = raw.seed_answer.strip()
        seed_origin = raw.seed_origin
        if not seed_answer:
            matched_item, _ = best_answer_match(raw.question, answer_bank, threshold=0.72)
            if matched_item is not None:
                seed_answer = matched_item.answer.strip()
                seed_origin = f"近似题匹配：{matched_item.source_label}"

        matched_aggregate: QuestionAggregate | None = None
        best_score = 0.0
        for aggregate in aggregates:
            score = similarity_score(raw.question, aggregate.question)
            if score > best_score:
                best_score = score
                matched_aggregate = aggregate
        if matched_aggregate is None or best_score < 0.95:
            aggregate = QuestionAggregate(
                key=md5(normalize_text(raw.question).encode("utf-8")).hexdigest(),
                question=raw.question,
                seed_answer=seed_answer,
                seed_origin=seed_origin,
                primary_tag=infer_tag(raw.module, raw.section_title, raw.question),
                module=raw.module,
                platform=raw.platform,
            )
            aggregate.sources.append(
                SourceRef(
                    source_name=raw.source_name,
                    source_url=raw.source_url,
                    page_group=raw.page_group,
                    section_title=raw.section_title,
                    platform=raw.platform,
                    company=raw.company,
                    module=raw.module,
                )
            )
            aggregates.append(aggregate)
            continue

        matched_aggregate.sources.append(
            SourceRef(
                source_name=raw.source_name,
                source_url=raw.source_url,
                page_group=raw.page_group,
                section_title=raw.section_title,
                platform=raw.platform,
                company=raw.company,
                module=raw.module,
            )
        )
        if choose_better_seed(matched_aggregate.seed_answer, seed_answer):
            matched_aggregate.seed_answer = seed_answer
            matched_aggregate.seed_origin = seed_origin
        if matched_aggregate.primary_tag == "其他":
            matched_aggregate.primary_tag = infer_tag(raw.module, raw.section_title, raw.question)

    return aggregates


def build_generation_signature(aggregate: QuestionAggregate) -> str:
    seed_fragment = normalize_text(trim_text(aggregate.seed_answer, 600))
    return md5(f"{normalize_text(aggregate.question)}|{seed_fragment}|{aggregate.primary_tag}".encode("utf-8")).hexdigest()


def heuristic_basics(tag: str) -> list[str]:
    presets = {
        "LLM基础": ["先定义模型范式", "说明训练目标", "补结构与能力边界"],
        "Transformer结构": ["先讲模块组成", "再讲计算路径", "补复杂度与取舍"],
        "模型微调": ["说明微调目标", "比较参数效率方案", "补数据与显存约束"],
        "推理优化与部署": ["先讲性能瓶颈", "再讲优化手段", "补精度与成本权衡"],
        "训练并行与系统": ["先讲资源瓶颈", "再讲并行策略", "补通信与稳定性"],
        "RAG与向量检索": ["先讲检索链路", "再讲召回与重排", "补效果评测方式"],
        "Agent与工具调用": ["先讲规划与调用", "再讲状态维护", "补异常处理策略"],
        "后训练与对齐": ["先讲偏好目标", "再讲损失差异", "补稳定性与副作用"],
        "多模态与视觉语言": ["先讲模态对齐", "再讲编码器设计", "补训练与推理成本"],
        "生成模型与扩散": ["先讲噪声过程", "再讲网络结构", "补采样效率与质量"],
        "经典算法与编程": ["先讲时间复杂度", "再讲边界条件", "补工程实现细节"],
    }
    return presets.get(tag, ["先定义核心概念", "再解释关键机制", "最后补工程取舍"])


def build_seed_card(aggregate: QuestionAggregate) -> QuestionCard:
    detailed = clean_markdown_text(strip_html(aggregate.seed_answer or ""))
    if len(detailed) > 900:
        detailed = trim_text(detailed, 900)
    if len(detailed) < 120:
        detailed = (
            f"{detailed} 答题时还需要补充：这个问题涉及的核心模块、为什么这样设计、以及实际工程里可能遇到的性能、显存或数据质量约束。"
        ).strip()
    case_simulation = (
        f"面试表达可以这样组织：先用一句话回答“{aggregate.question}”的核心结论，"
        f"再按“原理 -> 适用场景 -> 工程代价”的顺序展开。若面试官继续追问，"
        "就结合你做过的训练、推理、评测或排障经历，说一个具体案例来支撑判断。"
    )
    return QuestionCard(
        basics=heuristic_basics(aggregate.primary_tag),
        detailed_answer=detailed,
        case_simulation=case_simulation,
        generated_from="seed",
    )


def generate_cards_batch(
    client: OpenAI | None,
    model_name: str,
    batch: list[QuestionAggregate],
) -> dict[str, QuestionCard]:
    if client is None:
        return {
            item.key: QuestionCard(
                basics=["建议补充定义", "建议补充公式或模块", "建议补充工程取舍"],
                detailed_answer="当前未配置可用 LLM，暂未生成详细解答。",
                case_simulation="当前未配置可用 LLM，暂未生成案例模拟。",
                generated_from="fallback",
            )
            for item in batch
        }

    payload = []
    for item in batch:
        payload.append(
            {
                "id": item.key,
                "question": item.question,
                "topic": item.primary_tag,
                "module": item.module,
                "reference_answer": trim_text(item.seed_answer, 900),
            }
        )

    system_prompt = (
        "你是资深 AI 算法面试官和知识库编辑。"
        "请把输入的一批面试题整理成网页资料卡。"
        "必须返回 JSON 数组，数组元素字段固定为："
        '[{"id":"题目id","basics":["点1","点2","点3"],"detailed_answer":"...","case_simulation":"..."}]。'
        "规则如下："
        "1. basics 必须正好 3 条，每条 10-28 个中文字符，补充基础概念、关键模块、公式或工程术语；"
        "2. detailed_answer 220-420 个中文字符，先给结论，再解释原理、对比、工程权衡；"
        "3. case_simulation 120-220 个中文字符，给一个面试追问回答示例，或一个项目/业务案例模拟；"
        "4. 如果给了 reference_answer，优先吸收其中信息，但要改写得清晰、完整、适合网页阅读；"
        "5. 不要遗漏任何题，不要输出额外说明。"
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.25,
        )
        parsed = parse_json_payload(response.choices[0].message.content)
        if not isinstance(parsed, list):
            raise ValueError("模型未返回 JSON 数组")
        cards: dict[str, QuestionCard] = {}
        for item in parsed:
            if not isinstance(item, dict) or "id" not in item:
                continue
            basics = item.get("basics") or []
            if not isinstance(basics, list):
                basics = []
            basics = [clean_markdown_text(str(text)) for text in basics if clean_markdown_text(str(text))]
            while len(basics) < 3:
                basics.append("建议补充相关概念")
            cards[str(item["id"])] = QuestionCard(
                basics=basics[:3],
                detailed_answer=clean_markdown_text(str(item.get("detailed_answer", ""))),
                case_simulation=clean_markdown_text(str(item.get("case_simulation", ""))),
                generated_from="llm",
            )
        if len(cards) != len(batch):
            raise ValueError("模型返回题卡数量不完整")
        return cards
    except Exception:
        if len(batch) == 1:
            item = batch[0]
            return {
                item.key: QuestionCard(
                    basics=["先定义核心概念", "再解释关键机制", "最后补工程取舍"],
                    detailed_answer=(
                        f"这道题建议先给一句结论：{item.question} 的回答要围绕定义、核心原理和工程权衡展开。"
                        "如果你有项目经验，可以补充自己实际做过的方案、踩过的坑和最终指标变化；"
                        "如果没有项目经验，就用模型结构、训练目标、推理成本、效果收益这四个角度来组织回答。"
                    ),
                    case_simulation=(
                        "面试表达可以这样收尾：先说清楚这个方法解决了什么问题，再补一句它的代价是什么，"
                        "最后用“如果让我在项目里落地，我会先做小规模验证”来体现工程意识。"
                    ),
                    generated_from="fallback",
                )
            }

    cards: dict[str, QuestionCard] = {}
    for sub_batch in chunk_list(batch, 1):
        cards.update(generate_cards_batch(client, model_name, sub_batch))
    return cards


def enrich_question_cards(
    aggregates: list[QuestionAggregate],
    cache: dict[str, dict],
    client: OpenAI | None,
    model_name: str,
    cache_path: Path | None = None,
) -> dict[str, QuestionCard]:
    card_map: dict[str, QuestionCard] = {}
    pending: list[QuestionAggregate] = []
    batch_size = 20

    for aggregate in aggregates:
        signature = build_generation_signature(aggregate)
        cached = cache.get(aggregate.key)
        if cached and cached.get("signature") == signature:
            card_map[aggregate.key] = QuestionCard(
                basics=cached["basics"],
                detailed_answer=cached["detailed_answer"],
                case_simulation=cached["case_simulation"],
                generated_from=cached.get("generated_from", "cache"),
            )
        elif aggregate.seed_answer.strip():
            card = build_seed_card(aggregate)
            card_map[aggregate.key] = card
            cache[aggregate.key] = {
                "question": aggregate.question,
                "signature": signature,
                "basics": card.basics,
                "detailed_answer": card.detailed_answer,
                "case_simulation": card.case_simulation,
                "generated_from": card.generated_from,
            }
        else:
            pending.append(aggregate)

    total_batches = len(chunk_list(pending, batch_size))
    for batch_index, batch in enumerate(chunk_list(pending, batch_size), start=1):
        print(f"[*] 生成题卡批次 {batch_index}/{total_batches}，本批 {len(batch)} 题")
        generated_cards = generate_cards_batch(client, model_name, batch)
        for aggregate in batch:
            card = generated_cards[aggregate.key]
            card_map[aggregate.key] = card
            cache[aggregate.key] = {
                "question": aggregate.question,
                "signature": build_generation_signature(aggregate),
                "basics": card.basics,
                "detailed_answer": card.detailed_answer,
                "case_simulation": card.case_simulation,
                "generated_from": card.generated_from,
            }
        if cache_path is not None:
            save_card_cache(cache_path, cache)

    return card_map


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = sanitize_markdown_html(content.strip()) + "\n"
    path.write_text(sanitized, encoding="utf-8", newline="\n")


def render_source_links(sources: list[SourceRef]) -> list[str]:
    unique_items: list[tuple[str, str]] = []
    seen = set()
    for source in sources:
        label = f"{source.source_name} / {source.section_title} / {source.company or '未知'}"
        key = (label, source.source_url)
        if key in seen:
            continue
        seen.add(key)
        unique_items.append(key)
    lines: list[str] = []
    for label, url in unique_items[:5]:
        if url:
            lines.append(f"- 来源：[{label}]({url})")
        else:
            lines.append(f"- 来源：{label}")
    if len(unique_items) > 5:
        lines.append(f"- 其余来源：还有 {len(unique_items) - 5} 条相似题来源")
    return lines


def render_card(index: int, aggregate: QuestionAggregate, card: QuestionCard) -> str:
    lines = [
        f"### {index}. {aggregate.question}",
        "",
        f"- 主标签：{aggregate.primary_tag}",
        f"- 来源条数：{len(aggregate.sources)}",
        f"- 答案生成方式：{aggregate.seed_origin or '模型自动补全'}",
    ]
    lines.extend(render_source_links(aggregate.sources))
    lines.extend(["", "### 基础知识补充", ""])
    for point in card.basics:
        lines.append(f"- {point}")
    lines.extend(
        [
            "",
            "### 详细解答",
            "",
            card.detailed_answer,
            "",
            "### 案例模拟",
            "",
            card.case_simulation,
            "",
        ]
    )
    return "\n".join(lines)


def render_source_page(
    title: str,
    intro_lines: list[str],
    grouped_questions: list[tuple[str, list[QuestionAggregate]]],
    card_map: dict[str, QuestionCard],
) -> str:
    lines = [f"# {title}", ""]
    lines.extend(intro_lines)
    lines.append("")
    counter = 1
    for section_title, questions in grouped_questions:
        if section_title:
            lines.extend([f"## {section_title}", ""])
        for aggregate in questions:
            lines.append(render_card(counter, aggregate, card_map[aggregate.key]))
            counter += 1
    return "\n".join(lines)


def build_page_groups(
    raw_questions: list[RawQuestion],
    aggregate_map: dict[str, QuestionAggregate],
) -> dict[tuple[str, str], dict[str, list[QuestionAggregate]]]:
    page_groups: dict[tuple[str, str], dict[str, list[QuestionAggregate]]] = defaultdict(lambda: defaultdict(list))
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)

    for raw in raw_questions:
        key = md5(normalize_text(raw.question).encode("utf-8")).hexdigest()
        if key not in aggregate_map:
            for aggregate in aggregate_map.values():
                if similarity_score(raw.question, aggregate.question) >= 0.95:
                    key = aggregate.key
                    break
        if key not in aggregate_map:
            continue
        group_key = (raw.source_name, raw.page_group)
        if key in seen[group_key]:
            continue
        seen[group_key].add(key)
        page_groups[group_key][raw.section_title].append(aggregate_map[key])

    return page_groups


def render_overview(
    raw_questions: list[RawQuestion],
    aggregates: list[QuestionAggregate],
    external_results: list[ExternalFetchResult],
    source_page_meta: list[tuple[str, int]],
    merged_page_meta: list[tuple[str, int]],
) -> str:
    source_counter = Counter(item.source_name for item in raw_questions)
    platform_counter = Counter(item.platform for item in raw_questions)
    lines = [
        "# 全自动面经资料卡总览",
        "",
        f"> 自动生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 原始题目总数：{len(raw_questions)}",
        f"> 去重后题目总数：{len(aggregates)}",
        "",
        "## 本次导入策略",
        "",
        "- 两个 GitHub 仓库统一抽题，并把已有答案转成标准化网页题卡。",
        "- 对没有现成答案的题目，优先做近似题匹配，再由大模型补全“基础知识补充 + 详细解答 + 案例模拟”。",
        "- 真实面经 URL 和 GitHub 题库进入同一条流水线，统一去重、统一生成网页页面。",
        "",
        "## 数据分布",
        "",
    ]
    for source_name, count in source_counter.items():
        lines.append(f"- 来源：{source_name} | {count} 道原始题")
    lines.append("")
    for platform, count in platform_counter.items():
        lines.append(f"- 平台：{platform} | {count} 道原始题")
    lines.extend(["", "## 真实面经抓取状态", ""])
    if external_results:
        for result in external_results:
            lines.append(
                f"- {result.platform} / {result.company} / {result.title} | {result.status} | "
                f"{result.question_count} 题 | {result.message}"
            )
    else:
        lines.append("- 当前未配置外部 URL 来源。")
    lines.extend(["", "## 网页产物", ""])
    for title, count in source_page_meta:
        lines.append(f"- 来源页：{title} | {count} 道题")
    lines.extend(["", "## 融合题库页", ""])
    for title, count in merged_page_meta:
        lines.append(f"- 模块页：{title} | {count} 道题")
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

    repo1_raw = parse_repo1_raw_questions(repo1_qas)
    repo2_sections = parse_repo2_sections(repo2_root)
    repo2_raw = parse_repo2_raw_questions(repo2_sections, answer_bank)
    print(f"[*] GitHub 仓1抽取 {len(repo1_raw)} 道题，仓2抽取 {len(repo2_raw)} 道题")

    external_specs = load_external_sources(base_dir / "interview_source_urls.yaml")
    external_raw, external_results = fetch_external_sources(external_specs)
    print(f"[*] 外部真实面经抽取 {len(external_raw)} 道题，来源条目 {len(external_results)} 个")

    all_raw_questions = repo1_raw + repo2_raw + external_raw
    aggregates = merge_questions(all_raw_questions, answer_bank)
    print(f"[*] 合并前 {len(all_raw_questions)} 道题，去重后 {len(aggregates)} 道题")
    aggregate_map = {aggregate.key: aggregate for aggregate in aggregates}

    cache_path = base_dir / "archive" / "interview_cards" / "qa_cards_cache.json"
    cache = load_card_cache(cache_path)
    card_map = enrich_question_cards(aggregates, cache, client, model_name, cache_path)
    save_card_cache(cache_path, cache)

    output_dir = base_dir / "reports" / "专题题库" / "全自动面经资料卡"
    date_prefix = datetime.now().strftime("%Y%m%d")

    page_groups = build_page_groups(all_raw_questions, aggregate_map)
    source_page_meta: list[tuple[str, int]] = []
    for (source_name, page_group), sections in sorted(page_groups.items(), key=lambda item: item[0]):
        grouped_questions = [(section_title, questions) for section_title, questions in sections.items()]
        flat_count = sum(len(questions) for _, questions in grouped_questions)
        intro_lines = [
            f"> 来源分组：{source_name}",
            f"> 本页题目数：{flat_count}",
            "> 每题均包含基础知识补充、详细解答和案例模拟。",
        ]
        page_title = f"{source_name}：{page_group}"
        output_path = output_dir / f"{date_prefix}_{safe_name(page_title)}.md"
        write_text(output_path, render_source_page(page_title, intro_lines, grouped_questions, card_map))
        source_page_meta.append((page_title, flat_count))

    merged_page_meta: list[tuple[str, int]] = []
    by_tag: dict[str, list[QuestionAggregate]] = defaultdict(list)
    for aggregate in aggregates:
        by_tag[aggregate.primary_tag].append(aggregate)
    for tag, items in sorted(by_tag.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        ordered_items = sorted(items, key=lambda item: (-len(item.sources), item.question))
        page_title = f"融合题库：{tag}"
        intro_lines = [
            "> 已经把 GitHub 题库和真实面经合并去重。",
            f"> 本页共 {len(ordered_items)} 道题，按同题合并后的题卡展示。",
        ]
        output_path = output_dir / f"{date_prefix}_{safe_name(page_title)}.md"
        write_text(
            output_path,
            render_source_page(page_title, intro_lines, [("", ordered_items)], card_map),
        )
        merged_page_meta.append((page_title, len(ordered_items)))

    overview_path = output_dir / f"{date_prefix}_全自动面经资料卡总览.md"
    write_text(
        overview_path,
        render_overview(
            all_raw_questions,
            aggregates,
            external_results,
            source_page_meta,
            merged_page_meta,
        ),
    )

    print(f"[√] 已生成全自动面经资料卡，输出目录：{output_dir}")


if __name__ == "__main__":
    main()
