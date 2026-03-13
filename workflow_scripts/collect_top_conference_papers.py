from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests
import yaml


OPENREVIEW_API_URL = "https://api2.openreview.net/notes"
DEFAULT_HEADERS = {
    "User-Agent": "llm-interview-auto-fetch/3.0 (+https://github.com/HuangShengZeBlueSky/llm_interview_auto_fetch)"
}


@dataclass(frozen=True)
class VenueConfig:
    invitation: str
    accepted_venueid: str
    venue_label: str


@dataclass(frozen=True)
class PaperRecord:
    title: str
    authors: tuple[str, ...]
    keywords: tuple[str, ...]
    abstract: str
    forum_url: str
    pdf_url: str
    venue_label: str
    avg_rating: float
    review_count: int
    module: str


VENUES = [
    VenueConfig(
        invitation="ICLR.cc/2026/Conference/-/Submission",
        accepted_venueid="ICLR.cc/2026/Conference",
        venue_label="ICLR 2026",
    ),
    VenueConfig(
        invitation="NeurIPS.cc/2025/Conference/-/Submission",
        accepted_venueid="NeurIPS.cc/2025/Conference",
        venue_label="NeurIPS 2025",
    ),
    VenueConfig(
        invitation="ICML.cc/2025/Conference/-/Submission",
        accepted_venueid="ICML.cc/2025/Conference",
        venue_label="ICML 2025",
    ),
]


MODULE_RULES = [
    ("Agents & Tool Use", ["agent", "tool", "planner", "planning", "workflow", "browser", "mcp"]),
    ("Alignment & Post-training", ["alignment", "preference", "reward", "dpo", "ppo", "rlhf", "post-training"]),
    ("Multimodal Models", ["multimodal", "vision-language", "vlm", "video", "audio", "speech", "image-text"]),
    ("Reasoning & CoT", ["reasoning", "chain-of-thought", "cot", "verifier", "deliberation", "search"]),
    ("RAG & Knowledge Editing", ["retrieval", "rag", "knowledge", "editing", "memory", "grounding"]),
    ("Efficiency & Systems", ["quantization", "compression", "distill", "kv cache", "attention", "serving", "mixture-of-experts", "moe", "inference", "system"]),
    ("Evaluation & Benchmarks", ["benchmark", "evaluation", "judge", "dataset", "leaderboard"]),
    ("Diffusion & Generation", ["diffusion", "generation", "generative", "denoising"]),
    ("AI for Science", ["protein", "molecule", "biology", "chemistry", "material", "science", "drug"]),
    ("World Models & Robotics", ["robot", "robotics", "world model", "control", "policy", "embodied"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总顶会 accepted 高分论文并按模块生成页面")
    parser.add_argument(
        "--reuse-manifest",
        action="store_true",
        help="如果当天 manifest 已存在，则直接复用它生成页面，避免重复联网抓取",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def unwrap(value):
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def parse_retry_seconds(response: requests.Response, attempt: int) -> int:
    try:
        payload = response.json()
        message = payload.get("message", "")
        matched = re.search(r"try again in (\d+) seconds", message, re.IGNORECASE)
        if matched:
            return max(int(matched.group(1)), 1)
    except ValueError:
        pass
    return min(2 ** attempt, 30)


def request_openreview_notes(params: dict, max_retries: int = 8) -> dict:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.get(
                OPENREVIEW_API_URL,
                params=params,
                headers=DEFAULT_HEADERS,
                timeout=(30, 180),
            )
            if response.status_code != 429:
                response.raise_for_status()
                return response.json()

            wait_seconds = parse_retry_seconds(response, attempt)
            print(f"[429] OpenReview 限流，{wait_seconds}s 后重试...")
            time.sleep(wait_seconds)
        except requests.RequestException as exc:
            last_error = exc
            wait_seconds = min(2 ** attempt, 30)
            print(f"[retry] OpenReview 请求失败：{exc}，{wait_seconds}s 后重试...")
            time.sleep(wait_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError("OpenReview 请求失败，且没有返回可用响应。")


def iter_submissions(venue: VenueConfig, page_size: int = 50):
    offset = 0
    while True:
        payload = request_openreview_notes(
            {
                "invitation": venue.invitation,
                "content.venueid": venue.accepted_venueid,
                "limit": page_size,
                "offset": offset,
                "details": "directReplies",
            }
        )
        notes = payload.get("notes", [])
        if not notes:
            break

        for note in notes:
            yield note

        offset += len(notes)
        if len(notes) < page_size:
            break


def extract_ratings(note: dict) -> list[float]:
    ratings: list[float] = []
    direct_replies = note.get("details", {}).get("directReplies", [])
    for reply in direct_replies:
        invitations = reply.get("invitations", [])
        if not any("Official_Review" in invitation for invitation in invitations):
            continue

        content = reply.get("content", {})
        candidates = [
            unwrap(content.get("rating")),
            unwrap(content.get("overall_recommendation")),
            unwrap(content.get("recommendation")),
        ]
        for rating in candidates:
            if isinstance(rating, (int, float)):
                ratings.append(float(rating))
                break
            if isinstance(rating, str):
                matched = re.search(r"(\d+(?:\.\d+)?)", rating)
                if matched:
                    ratings.append(float(matched.group(1)))
                    break
    return ratings


def classify_module(title: str, abstract: str, keywords: tuple[str, ...]) -> str:
    haystack = " ".join([title, abstract, " ".join(keywords)]).lower()
    for module, keywords_list in MODULE_RULES:
        if any(keyword in haystack for keyword in keywords_list):
            return module
    return "Reasoning & CoT"


def build_record(note: dict, venue_label: str) -> PaperRecord | None:
    ratings = extract_ratings(note)
    if len(ratings) < 3:
        return None

    content = note.get("content", {})
    title = unwrap(content.get("title")) or "Untitled"
    authors = tuple(unwrap(content.get("authors")) or [])
    keywords = tuple(str(item) for item in (unwrap(content.get("keywords")) or []))
    abstract = unwrap(content.get("abstract")) or ""
    forum = note.get("forum")
    pdf_path = unwrap(content.get("pdf"))
    module = classify_module(title, abstract, keywords)

    return PaperRecord(
        title=title,
        authors=authors,
        keywords=keywords,
        abstract=abstract,
        forum_url=f"https://openreview.net/forum?id={forum}" if forum else "",
        pdf_url=f"https://openreview.net{pdf_path}" if pdf_path else "",
        venue_label=venue_label,
        avg_rating=sum(ratings) / len(ratings),
        review_count=len(ratings),
        module=module,
    )


def safe_segment(text: str) -> str:
    cleaned = str(text).strip().strip(".")
    cleaned = cleaned.replace("&", " and ")
    cleaned = cleaned.replace("：", "-").replace(":", "-")
    cleaned = re.sub(r'[<>"/\\|?*]+', "", cleaned)
    cleaned = re.sub(r"\s+", "-", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "untitled"


def render_overview(records: list[PaperRecord], links_by_module: dict[str, str]) -> str:
    by_venue: dict[str, list[PaperRecord]] = defaultdict(list)
    by_module: dict[str, list[PaperRecord]] = defaultdict(list)
    for record in records:
        by_venue[record.venue_label].append(record)
        by_module[record.module].append(record)

    venue_lines = [
        "| 会议 | 入选论文数 | 最高平均分 |",
        "| --- | ---: | ---: |",
    ]
    for venue_label, items in sorted(by_venue.items()):
        venue_lines.append(f"| {venue_label} | {len(items)} | {max(item.avg_rating for item in items):.2f} |")

    module_lines = [
        "| 模块 | 论文数 | 模块入口 |",
        "| --- | ---: | --- |",
    ]
    for module, items in sorted(by_module.items()):
        module_lines.append(f"| {module} | {len(items)} | [查看模块汇总]({links_by_module[module]}) |")

    top_rows = [
        "| 排名 | 标题 | 会议 | 平均分 | Review 数 | 模块 |",
        "| ---: | --- | --- | ---: | ---: | --- |",
    ]
    for index, record in enumerate(records, start=1):
        title = f"[{record.title}]({record.forum_url})" if record.forum_url else record.title
        top_rows.append(
            f"| {index} | {title} | {record.venue_label} | {record.avg_rating:.2f} | {record.review_count} | {record.module} |"
        )

    lines = [
        "# 顶会高分论文总览",
        "",
        f"> 自动生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "> 采集范围：ICLR 2026、NeurIPS 2025、ICML 2025 的 accepted 论文",
        "> 入选规则：平均评分排序，且至少 3 个官方 review；优先保留每个会议的高分论文，再按模块聚合",
        "> 说明：不同会议的评分量表可能不同，因此本页更适合做“分会议筛选”和“按模块追踪”，不建议直接跨会议比较原始分数。",
        "",
        "## 会议覆盖",
        "",
        "\n".join(venue_lines),
        "",
        "## 模块覆盖",
        "",
        "\n".join(module_lines),
        "",
        "## 分会议高分论文列表",
        "",
        "\n".join(top_rows),
        "",
    ]

    for venue_label, items in sorted(by_venue.items()):
        lines.append(f"## {venue_label}")
        lines.append("")
        for record in items:
            title = f"[{record.title}]({record.forum_url})" if record.forum_url else record.title
            lines.append(
                f"- {title} | 平均分 {record.avg_rating:.2f} | {record.review_count} reviews | {record.module}"
            )
        lines.append("")

    return "\n".join(lines)


def trim_abstract(text: str, limit: int = 260) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def render_module_page(module: str, records: list[PaperRecord], overview_link: str) -> str:
    lines = [
        f"# 顶会高分论文模块汇总：{module}",
        "",
        f"> 自动生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 返回总览：[顶会高分论文总览]({overview_link})",
        "> 说明：不同会议评分量表可能不同，因此这里更适合做候选论文池，而不是做绝对分数排名。",
        "",
    ]

    for index, record in enumerate(records, start=1):
        lines.append(f"## {index}. {record.title}")
        lines.append("")
        lines.append(f"- 会议：{record.venue_label}")
        lines.append(f"- 平均评分：{record.avg_rating:.2f}")
        lines.append(f"- Review 数：{record.review_count}")
        lines.append(f"- 作者：{', '.join(record.authors[:8]) or 'N/A'}")
        lines.append(f"- 关键词：{', '.join(record.keywords[:8]) or 'N/A'}")
        lines.append(f"- 摘要速览：{trim_abstract(record.abstract)}")
        if record.forum_url:
            lines.append(f"- OpenReview：{record.forum_url}")
        if record.pdf_url:
            lines.append(f"- PDF：{record.pdf_url}")
        lines.append("")

    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8", newline="\n")


def write_manifest(manifest_path: Path, records: list[PaperRecord]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "records": [record.__dict__ for record in records],
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_records_from_manifest(manifest_path: Path) -> list[PaperRecord]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for item in payload.get("records", []):
        records.append(
            PaperRecord(
                title=item["title"],
                authors=tuple(item.get("authors", [])),
                keywords=tuple(item.get("keywords", [])),
                abstract=item.get("abstract", ""),
                forum_url=item.get("forum_url", ""),
                pdf_url=item.get("pdf_url", ""),
                venue_label=item["venue_label"],
                avg_rating=float(item["avg_rating"]),
                review_count=int(item["review_count"]),
                module=item["module"],
            )
        )
    return records


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parents[1]
    config = load_config(base_dir / "config.yaml")
    paths = config.get("paths", {})
    reports_dir = base_dir / paths.get("reports", "reports")
    archive_dir = base_dir / paths.get("archive", "archive")
    date_prefix = datetime.now().strftime("%Y%m%d")
    manifest_path = archive_dir / "papers" / "_manifests" / f"top_papers_{date_prefix}.json"

    if args.reuse_manifest and manifest_path.exists():
        selected_records = load_records_from_manifest(manifest_path)
        print(f"[*] 已复用 manifest：{manifest_path}")
    else:
        print("[*] 正在拉取顶会 accepted 高分论文并按模块汇总...")
        selected_records = []
        per_venue_limit = 8

        for venue in VENUES:
            venue_records = []
            for note in iter_submissions(venue):
                record = build_record(note, venue.venue_label)
                if record is None:
                    continue
                venue_records.append(record)

            venue_records.sort(key=lambda item: (item.avg_rating, item.review_count, item.title), reverse=True)
            selected = venue_records[:per_venue_limit]
            selected_records.extend(selected)
            print(f"    [√] {venue.venue_label}: 命中 {len(selected)} 篇高分论文")

        write_manifest(manifest_path, selected_records)
        print(f"[√] 总计写入 {len(selected_records)} 篇论文，manifest: {manifest_path}")

    selected_records.sort(key=lambda item: (item.venue_label, -item.avg_rating, -item.review_count, item.title))

    overview_group = "高分论文索引"
    overview_name = f"{date_prefix}_顶会高分论文总览.md"
    overview_path = reports_dir / "论文精读" / overview_group / overview_name
    overview_link = (
        f"/reports/{safe_segment('论文精读')}/{safe_segment(overview_group)}/{safe_segment(overview_name[:-3])}"
    )

    by_module: dict[str, list[PaperRecord]] = defaultdict(list)
    for record in selected_records:
        by_module[record.module].append(record)

    links_by_module: dict[str, str] = {}
    for module in sorted(by_module):
        filename = f"{date_prefix}_顶会高分论文_{module}.md"
        links_by_module[module] = (
            f"/reports/{safe_segment('论文精读')}/{safe_segment(module)}/{safe_segment(filename[:-3])}"
        )

    write_text(overview_path, render_overview(selected_records, links_by_module))

    for module, items in sorted(by_module.items()):
        items.sort(key=lambda item: (item.venue_label, -item.avg_rating, -item.review_count, item.title))
        filename = f"{date_prefix}_顶会高分论文_{module}.md"
        module_path = reports_dir / "论文精读" / module / filename
        write_text(module_path, render_module_page(module, items, overview_link))
    print(f"[√] 已更新论文页面，共 {len(selected_records)} 篇。")


if __name__ == "__main__":
    main()
