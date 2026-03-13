# -*- coding: utf-8 -*-
"""
从 OpenReview 拉取已接收论文，并按平均评分筛选后写入 raw_data_papers。

示例：
python workflow_scripts/collect_openreview.py ^
  --invitation ICLR.cc/2026/Conference/-/Submission ^
  --accepted-venueid ICLR.cc/2026/Conference ^
  --min-avg-rating 7.5 ^
  --max-results 20
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
import yaml

OPENREVIEW_API_URL = "https://api2.openreview.net/notes"
DEFAULT_HEADERS = {
    "User-Agent": "llm-interview-auto-fetch/2.1 (+https://github.com/HuangShengZeBlueSky/llm_interview_auto_fetch)"
}


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="拉取 OpenReview 已接收论文并写入 raw_data_papers")
    parser.add_argument(
        "--invitation",
        default="ICLR.cc/2026/Conference/-/Submission",
        help="OpenReview submission invitation",
    )
    parser.add_argument(
        "--accepted-venueid",
        default="ICLR.cc/2026/Conference",
        help="已接收论文对应的 venueid；ICLR 2026 可直接用主会 venueid",
    )
    parser.add_argument(
        "--venue-label",
        default="ICLR 2026",
        help="写入原始文本时展示给人看的会议名",
    )
    parser.add_argument(
        "--min-avg-rating",
        type=float,
        default=7.5,
        help="平均评分阈值，只保留不低于该阈值的论文",
    )
    parser.add_argument(
        "--min-review-count",
        type=int,
        default=3,
        help="最少有效 review 数量",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=20,
        help="最多写入多少篇论文",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=50,
        help="每次请求 OpenReview 的分页大小",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="最多抓多少页；0 表示抓完全部 accepted 结果",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="覆盖默认输出目录；默认写入 config.yaml 中的 raw_data_papers",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印筛选结果，不写文件",
    )
    return parser.parse_args()


def unwrap(value):
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def parse_retry_seconds(response: requests.Response, attempt: int) -> int:
    try:
        payload = response.json()
        message = payload.get("message", "")
        match = re.search(r"try again in (\d+) seconds", message, re.IGNORECASE)
        if match:
            return max(int(match.group(1)), 1)
    except ValueError:
        pass
    return min(2 ** attempt, 30)


def request_openreview_notes(params: dict, max_retries: int = 6) -> dict:
    for attempt in range(max_retries):
        response = requests.get(
            OPENREVIEW_API_URL,
            params=params,
            headers=DEFAULT_HEADERS,
            timeout=120,
        )
        if response.status_code != 429:
            response.raise_for_status()
            return response.json()

        wait_seconds = parse_retry_seconds(response, attempt)
        print(f"[429] OpenReview 限流，{wait_seconds}s 后重试...")
        time.sleep(wait_seconds)

    response.raise_for_status()
    return {}


def iter_submissions(
    invitation: str,
    accepted_venueid: str,
    page_size: int,
    max_pages: int,
) -> Iterable[dict]:
    offset = 0
    page_index = 0
    while True:
        if max_pages and page_index >= max_pages:
            break

        payload = request_openreview_notes(
            {
                "invitation": invitation,
                "content.venueid": accepted_venueid,
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
        page_index += 1
        if len(notes) < page_size:
            break


def extract_ratings(note: dict) -> List[float]:
    ratings: List[float] = []
    direct_replies = note.get("details", {}).get("directReplies", [])
    for reply in direct_replies:
        invitations = reply.get("invitations", [])
        if not any("Official_Review" in invitation for invitation in invitations):
            continue

        rating = unwrap(reply.get("content", {}).get("rating"))
        if isinstance(rating, (int, float)):
            ratings.append(float(rating))
            continue
        if isinstance(rating, str):
            match = re.search(r"(\d+(?:\.\d+)?)", rating)
            if match:
                ratings.append(float(match.group(1)))
    return ratings


def safe_filename(text: str, limit: int = 80) -> str:
    cleaned = re.sub(r'[\\/*?:"<>|]+', "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:limit] or "untitled"


def format_list(values: Optional[List[str]]) -> str:
    if not values:
        return "N/A"
    return ", ".join(str(value) for value in values)


def build_record(note: dict, venue_label: str) -> Optional[Dict[str, object]]:
    ratings = extract_ratings(note)
    if not ratings:
        return None

    content = note.get("content", {})
    forum = note.get("forum")
    pdf_path = unwrap(content.get("pdf"))
    return {
        "title": unwrap(content.get("title")) or "Untitled",
        "authors": unwrap(content.get("authors")) or [],
        "keywords": unwrap(content.get("keywords")) or [],
        "abstract": unwrap(content.get("abstract")) or "",
        "tldr": unwrap(content.get("TLDR")) or "",
        "forum_id": forum,
        "forum_url": f"https://openreview.net/forum?id={forum}" if forum else "",
        "pdf_url": f"https://openreview.net{pdf_path}" if pdf_path else "",
        "venue_label": venue_label,
        "venueid": unwrap(content.get("venueid")) or "",
        "ratings": ratings,
        "avg_rating": sum(ratings) / len(ratings),
        "review_count": len(ratings),
    }


def write_record(output_dir: Path, manifest_path: Path, record: Dict[str, object], index: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = safe_filename(str(record["title"]))
    output_path = output_dir / f"openreview_{timestamp}_{index:03d}_{safe_title}.txt"

    content = f"""【来源】: OpenReview
【会议】: {record['venue_label']}
【接收状态】: Accepted
【平均评分】: {record['avg_rating']:.2f}
【评分列表】: {record['ratings']}
【Review 数量】: {record['review_count']}
【标题】: {record['title']}
【作者】: {format_list(record['authors'])}
【关键词】: {format_list(record['keywords'])}
【论坛】: {record['forum_url']}
【PDF】: {record['pdf_url']}

【TL;DR】:
{record['tldr'] or 'N/A'}

【摘要】:
{record['abstract'] or 'N/A'}
"""
    output_path.write_text(content, encoding="utf-8")

    with manifest_path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path


def main():
    args = parse_args()
    config = load_config()
    base_dir = Path(__file__).resolve().parents[1]
    paths = config.get("paths", {})

    output_dir = Path(args.output_dir) if args.output_dir else base_dir / paths.get("raw_data_papers", "raw_data_papers")
    manifest_dir = base_dir / paths.get("archive", "archive") / "papers" / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[*] 正在从 OpenReview 拉取 {args.venue_label} Accepted 论文，"
        f"筛选条件：avg_rating >= {args.min_avg_rating}, review_count >= {args.min_review_count}"
    )

    records: List[Dict[str, object]] = []
    for note in iter_submissions(
        args.invitation,
        args.accepted_venueid,
        args.page_size,
        args.max_pages,
    ):
        record = build_record(note, args.venue_label)
        if not record:
            continue
        if record["avg_rating"] < args.min_avg_rating:
            continue
        if record["review_count"] < args.min_review_count:
            continue
        records.append(record)

    records.sort(
        key=lambda item: (item["avg_rating"], item["review_count"], item["title"]),
        reverse=True,
    )
    records = records[: args.max_results]

    if not records:
        print("[-] 没有命中筛选结果。请降低评分阈值，或检查 invitation / venueid 是否正确。")
        return

    print(f"[*] 命中 {len(records)} 篇论文：")
    for index, record in enumerate(records, start=1):
        print(
            f"    {index:02d}. {record['avg_rating']:.2f} / {record['review_count']} reviews / {record['title']}"
        )

    if args.dry_run:
        print("[*] dry-run 模式，不写入 raw_data_papers。")
        return

    manifest_path = manifest_dir / (
        f"openreview_{safe_filename(args.venue_label.replace(' ', '_'), limit=40)}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )

    for index, record in enumerate(records, start=1):
        output_path = write_record(output_dir, manifest_path, record, index)
        print(f"    [√] 已写入: {output_path}")

    print(f"[*] 采集完成。manifest: {manifest_path}")


if __name__ == "__main__":
    main()
