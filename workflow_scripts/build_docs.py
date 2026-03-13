# -*- coding: utf-8 -*-
"""
生成 VitePress 静态网站配置文件及专题入口页。
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ReportItem:
    kind: str
    group: str
    sub_group: str
    filename: str
    title: str
    link: str
    date_text: str
    sort_key: str


GROUP_PRIORITY = {
    "专题题库": -20,
    "高分论文索引": -20,
}


def derive_title(md_file: str) -> str:
    stem = md_file[:-3] if md_file.endswith(".md") else md_file
    parts = stem.split("_", 2)
    if len(parts) >= 3:
        return parts[2] or stem
    if len(parts) == 2:
        return parts[1] or stem
    return stem


def extract_date_text(md_file: str) -> str:
    stem = md_file[:-3] if md_file.endswith(".md") else md_file
    match = re.match(r"^(\d{8})(?:_(\d{4,6}))?", stem)
    if not match:
        return "未知时间"

    day = match.group(1)
    time_part = match.group(2) or ""
    try:
        if len(time_part) == 6:
            return datetime.strptime(day + time_part, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
        if len(time_part) == 4:
            return datetime.strptime(day + time_part, "%Y%m%d%H%M").strftime("%Y-%m-%d %H:%M")
        return datetime.strptime(day, "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        return day


def build_link(*parts: str) -> str:
    return "/reports/" + "/".join(part.strip("/\\") for part in parts)


def safe_segment(text: str) -> str:
    cleaned = str(text).strip().strip(".")
    cleaned = cleaned.replace("&", " and ")
    cleaned = cleaned.replace("：", "-").replace(":", "-")
    cleaned = re.sub(r'[<>"/\\|?*]+', "", cleaned)
    cleaned = re.sub(r"\s+", "-", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "untitled"


def ensure_unique_filename(target_dir: str, filename: str) -> str:
    stem, ext = os.path.splitext(filename)
    candidate = filename
    index = 2
    while os.path.exists(os.path.join(target_dir, candidate)):
        candidate = f"{stem}-{index}{ext}"
        index += 1
    return candidate


def sync_reports_and_collect_items(reports_dir: str, docs_reports_dir: str) -> list[ReportItem]:
    items: list[ReportItem] = []

    for root, _, files in os.walk(reports_dir):
        relative_root = os.path.relpath(root, reports_dir)
        if relative_root == ".":
            continue

        path_parts = relative_root.split(os.sep)
        safe_path_parts = [safe_segment(part) for part in path_parts]
        target_dir = os.path.join(docs_reports_dir, *safe_path_parts)
        os.makedirs(target_dir, exist_ok=True)

        for md_file in sorted((name for name in files if name.endswith(".md")), reverse=True):
            title = derive_title(md_file)
            date_text = extract_date_text(md_file)
            safe_file = ensure_unique_filename(target_dir, f"{safe_segment(md_file[:-3])}.md")
            shutil.copy2(os.path.join(root, md_file), os.path.join(target_dir, safe_file))
            link = build_link(*safe_path_parts, safe_file[:-3])
            sort_key = md_file

            if path_parts[0] == "00_行业洞察":
                group = path_parts[1] if len(path_parts) > 1 else "未分类"
                items.append(
                    ReportItem(
                        kind="insights",
                        group=group,
                        sub_group="",
                        filename=md_file,
                        title=title,
                        link=link,
                        date_text=date_text,
                        sort_key=sort_key,
                    )
                )
            elif path_parts[0] == "论文精读":
                field_name = path_parts[1] if len(path_parts) > 1 else "其他"
                items.append(
                    ReportItem(
                        kind="papers",
                        group=field_name,
                        sub_group="",
                        filename=md_file,
                        title=title,
                        link=link,
                        date_text=date_text,
                        sort_key=sort_key,
                    )
                )
            elif path_parts[0] == "体系化课程":
                course_name = path_parts[1] if len(path_parts) > 1 else "未分类课程"
                items.append(
                    ReportItem(
                        kind="courses",
                        group=course_name,
                        sub_group="",
                        filename=md_file,
                        title=title,
                        link=link,
                        date_text=date_text,
                        sort_key=sort_key,
                    )
                )
            else:
                company = path_parts[0]
                tag = path_parts[1] if len(path_parts) > 1 else "其他"
                items.append(
                    ReportItem(
                        kind="interviews",
                        group=company,
                        sub_group=tag,
                        filename=md_file,
                        title=title,
                        link=link,
                        date_text=date_text,
                        sort_key=sort_key,
                    )
                )

    return sorted(items, key=lambda item: (item.sort_key, item.group, item.sub_group), reverse=True)


def group_items(items: list[ReportItem], key_func) -> dict[str, list[ReportItem]]:
    grouped: dict[str, list[ReportItem]] = defaultdict(list)
    for item in items:
        grouped[key_func(item)].append(item)
    return dict(sorted(grouped.items(), key=lambda pair: pair[0]))


def sorted_group_entries(grouped: dict[str, list[ReportItem]]) -> list[tuple[str, list[ReportItem]]]:
    return sorted(
        grouped.items(),
        key=lambda pair: (GROUP_PRIORITY.get(pair[0], 0), pair[0]),
    )


def find_featured_item(
    items: list[ReportItem],
    *,
    kind: str,
    group: str | None = None,
    title_contains: str | None = None,
) -> ReportItem | None:
    for item in items:
        if item.kind != kind:
            continue
        if group is not None and item.group != group:
            continue
        if title_contains is not None and title_contains not in item.title:
            continue
        return item
    return None


def write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content.strip() + "\n")


def render_item_list(items: list[ReportItem], with_prefix: str = "") -> str:
    if not items:
        return "- 暂无内容"
    lines = []
    for item in items:
        prefix = f"{with_prefix} " if with_prefix else ""
        lines.append(f"- {prefix}[{item.title}]({item.link}) | {item.date_text}")
    return "\n".join(lines)


def render_reports_home(items: list[ReportItem]) -> str:
    interviews_count = sum(1 for item in items if item.kind == "interviews")
    papers_count = sum(1 for item in items if item.kind == "papers")
    courses_count = sum(1 for item in items if item.kind == "courses")
    insights_count = sum(1 for item in items if item.kind == "insights")
    featured_interview = find_featured_item(items, kind="interviews", group="专题题库")
    featured_paper = find_featured_item(items, kind="papers", group="高分论文索引")

    featured_lines = []
    if featured_interview or featured_paper:
        featured_lines.extend(["## 精选入口", ""])
        if featured_interview:
            featured_lines.append(f"- [大厂 AI 高频题单]({featured_interview.link})")
        if featured_paper:
            featured_lines.append(f"- [顶会高分论文总览]({featured_paper.link})")
        featured_lines.append("")

    return f"""
# 知识库总览

这个页面是站点的总入口。当前内容已经拆成四个板块，后续你可以继续沿着这套结构扩充数据与页面能力。

## 板块导航

- [面经板块](./interviews/)：按公司、知识点浏览，适合做题与面试准备。
- [论文板块](./papers/)：按研究方向聚合，适合追前沿和做精读。
- [课程板块](./courses/)：按课程名聚合，支持持续搬运公开课与字幕笔记。
- [洞察板块](./insights/)：从最近内容中自动汇总趋势和风向。

## 当前内容规模

| 板块 | 数量 |
| :--- | ---: |
| 面经 | {interviews_count} |
| 论文 | {papers_count} |
| 课程 | {courses_count} |
| 洞察 | {insights_count} |

{chr(10).join(featured_lines).strip()}
"""


def render_interviews_by_tag(items: list[ReportItem]) -> str:
    lines = []
    for item in items:
        lines.append(f"- [{item.group}] [{item.title}]({item.link}) | {item.date_text}")
    return "\n".join(lines) or "- 暂无内容"


def render_interviews_page(items: list[ReportItem]) -> str:
    curated_items = [item for item in items if item.group == "专题题库"]
    regular_items = [item for item in items if item.group != "专题题库"]
    by_company = group_items(regular_items, lambda item: item.group)
    by_tag = group_items(items, lambda item: item.sub_group)

    lines = [
        "# 面经板块",
        "",
        "这里是面向求职准备的主战场。现在已经可以按公司和知识点双视角浏览，后续再叠加筛选、统计和题单就会很顺。",
        "",
    ]

    if curated_items:
        lines.extend(
            [
                "## 专题题库精选",
                "",
                render_item_list(curated_items),
                "",
            ]
        )

    lines.extend(
        [
        "## 按公司浏览",
        "",
        ]
    )

    for company, company_items in sorted_group_entries(by_company):
        lines.append(f"### {company} ({len(company_items)})")
        lines.append(render_item_list(company_items, with_prefix=f"[{company}]"))
        lines.append("")

    lines.extend(["## 按知识点浏览", ""])
    for tag, tag_items in sorted_group_entries(by_tag):
        lines.append(f"### {tag} ({len(tag_items)})")
        lines.append(render_interviews_by_tag(tag_items))
        lines.append("")

    return "\n".join(lines)


def render_papers_page(items: list[ReportItem]) -> str:
    curated_items = [item for item in items if item.group == "高分论文索引"]
    regular_items = [item for item in items if item.group != "高分论文索引"]
    by_field = group_items(regular_items, lambda item: item.group)
    lines = [
        "# 论文板块",
        "",
        "这里按研究方向聚合论文精读，已经符合你想要的“论文按领域分类”目标。",
        "",
    ]

    if curated_items:
        lines.extend(["## 顶会高分论文精选", "", render_item_list(curated_items), ""])

    for field_name, field_items in sorted_group_entries(by_field):
        lines.append(f"## {field_name} ({len(field_items)})")
        lines.append(render_item_list(field_items))
        lines.append("")

    return "\n".join(lines)


def render_courses_page(items: list[ReportItem]) -> str:
    by_course = group_items(items, lambda item: item.group)
    lines = [
        "# 课程板块",
        "",
        "这里按课程名聚合内容，适合持续搬运公开课、字幕稿和课程讲义。",
        "",
    ]

    for course_name, course_items in by_course.items():
        lines.append(f"## {course_name} ({len(course_items)})")
        lines.append(render_item_list(course_items))
        lines.append("")

    lines.extend(
        [
            "## 课程迁移接口",
            "",
            "你现在搬课程可以直接复用现有流水线，不需要额外改数据库：",
            "",
            "1. 把原始课程材料整理成 UTF-8 编码的 `.txt` 或 `.md` 文件，放到 `raw_data_courses/`。",
            "2. 文件名建议用 `课程名_章节名.txt`，例如 `CS231N_Lecture01_Intro.txt`。",
            "3. 如果想让分类更稳定，先把课程名补进根目录的 `taxonomy.yaml` -> `courses_tags`。",
            "4. 运行 `python workflow_scripts/process_course.py`，课程内容会被整理到 `reports/体系化课程/<课程名>/`。",
            "5. 再运行 `python workflow_scripts/build_docs.py` 和 `npm run docs:build`，站点会自动更新。",
            "",
            "如果你的来源是 YouTube/B 站视频，建议先做一层转写，把字幕导出成文本后再进这条通道。",
        ]
    )

    return "\n".join(lines)


def render_insights_page(items: list[ReportItem]) -> str:
    by_topic = group_items(items, lambda item: item.group)
    lines = [
        "# 洞察板块",
        "",
        "这部分来自最近内容的自动汇总，更适合做宏观复习和方向判断。",
        "",
    ]

    for topic, topic_items in by_topic.items():
        lines.append(f"## {topic} ({len(topic_items)})")
        lines.append(render_item_list(topic_items))
        lines.append("")

    return "\n".join(lines)


def render_sidebar_section(title: str, items: list[dict], collapsed: bool = False) -> dict:
    return {
        "text": title,
        "collapsed": collapsed,
        "items": items,
    }


def render_sidebar_report_items(items: list[ReportItem], prefix_builder=None) -> list[dict]:
    result = []
    for item in items:
        text = item.title if prefix_builder is None else prefix_builder(item)
        result.append({"text": text, "link": item.link})
    return result


def build_sidebar(items: list[ReportItem]) -> list[dict]:
    interviews = [item for item in items if item.kind == "interviews"]
    papers = [item for item in items if item.kind == "papers"]
    courses = [item for item in items if item.kind == "courses"]
    insights = [item for item in items if item.kind == "insights"]

    sidebar = [
        render_sidebar_section(
            "专题导航",
            [
                {"text": "知识库总览", "link": "/reports/"},
                {"text": "面经板块", "link": "/reports/interviews/"},
                {"text": "论文板块", "link": "/reports/papers/"},
                {"text": "课程板块", "link": "/reports/courses/"},
                {"text": "洞察板块", "link": "/reports/insights/"},
            ],
            collapsed=False,
        )
    ]

    by_company = group_items(interviews, lambda item: item.group)
    interview_items = []
    for company, company_items in sorted_group_entries(by_company):
        by_tag = group_items(company_items, lambda item: item.sub_group)
        interview_items.append(
            {
                "text": company,
                "collapsed": True,
                "items": [
                    {
                        "text": tag,
                        "collapsed": True,
                        "items": render_sidebar_report_items(tag_items),
                    }
                    for tag, tag_items in sorted_group_entries(by_tag)
                ],
            }
        )
    sidebar.append(render_sidebar_section("面经题库", interview_items, collapsed=False))

    by_field = group_items(papers, lambda item: item.group)
    paper_items = [
        {
            "text": field_name,
            "collapsed": True,
            "items": render_sidebar_report_items(field_items),
        }
        for field_name, field_items in sorted_group_entries(by_field)
    ]
    sidebar.append(render_sidebar_section("论文精读", paper_items, collapsed=False))

    by_course = group_items(courses, lambda item: item.group)
    course_items = [
        {
            "text": course_name,
            "collapsed": True,
            "items": render_sidebar_report_items(course_items_value),
        }
        for course_name, course_items_value in sorted_group_entries(by_course)
    ]
    sidebar.append(render_sidebar_section("课程笔记", course_items, collapsed=False))

    by_topic = group_items(insights, lambda item: item.group)
    insight_items = [
        {
            "text": topic,
            "collapsed": True,
            "items": render_sidebar_report_items(topic_items),
        }
        for topic, topic_items in sorted_group_entries(by_topic)
    ]
    sidebar.append(render_sidebar_section("行业洞察", insight_items, collapsed=False))
    return sidebar


def render_config(sidebar: list[dict]) -> str:
    return f"""
import {{ defineConfig }} from 'vitepress'

export default defineConfig({{
  title: "LLM 知识库",
  description: "面经、论文、课程三位一体的自动化知识库",
  base: "/llm_interview_auto_fetch/",
  themeConfig: {{
    nav: [
      {{ text: '首页', link: '/' }},
      {{ text: '面经', link: '/reports/interviews/' }},
      {{ text: '论文', link: '/reports/papers/' }},
      {{ text: '课程', link: '/reports/courses/' }},
      {{ text: '洞察', link: '/reports/insights/' }},
      {{ text: '仓库结构', link: '/repo-structure' }}
    ],
    sidebar: {{
      '/reports/': {json.dumps(sidebar, ensure_ascii=False, indent=6)}
    }},
    socialLinks: [
      {{ icon: 'github', link: 'https://github.com/HuangShengZeBlueSky/llm_interview_auto_fetch' }}
    ],
    search: {{
      provider: 'local'
    }}
  }}
}})
"""


def build() -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(base_dir, "reports")
    docs_dir = os.path.join(base_dir, "docs")
    docs_reports_dir = os.path.join(docs_dir, "reports")
    vitepress_dir = os.path.join(docs_dir, ".vitepress")

    os.makedirs(vitepress_dir, exist_ok=True)

    if os.path.exists(docs_reports_dir):
        shutil.rmtree(docs_reports_dir)
    os.makedirs(docs_reports_dir, exist_ok=True)

    items = sync_reports_and_collect_items(reports_dir, docs_reports_dir) if os.path.exists(reports_dir) else []

    write_text(os.path.join(docs_reports_dir, "index.md"), render_reports_home(items))
    write_text(
        os.path.join(docs_reports_dir, "interviews", "index.md"),
        render_interviews_page([item for item in items if item.kind == "interviews"]),
    )

    write_text(
        os.path.join(docs_reports_dir, "papers", "index.md"),
        render_papers_page([item for item in items if item.kind == "papers"]),
    )
    write_text(
        os.path.join(docs_reports_dir, "courses", "index.md"),
        render_courses_page([item for item in items if item.kind == "courses"]),
    )
    write_text(
        os.path.join(docs_reports_dir, "insights", "index.md"),
        render_insights_page([item for item in items if item.kind == "insights"]),
    )

    config_path = os.path.join(vitepress_dir, "config.mjs")
    write_text(config_path, render_config(build_sidebar(items)))
    print("[√] docs/.vitepress/config.mjs 与专题入口页已生成。")


if __name__ == "__main__":
    build()
