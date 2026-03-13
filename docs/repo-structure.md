# 仓库结构与采集方案

这个仓库现在本质上是一个三通道知识库流水线：

1. `面经通道`：接收截图、文本、PDF，输出结构化面经解析。
2. `论文通道`：接收 OpenReview / ArXiv / PDF / 文本，输出论文精读。
3. `课程通道`：接收字幕或笔记文本，输出教材级整理笔记。

## 目录地图

```text
大模型面试题库/
├─ raw_data/                    # 面经入口
├─ raw_data_papers/             # 论文入口
├─ raw_data_courses/            # 课程入口
├─ archive/                     # 原始资料归档
│  ├─ 字节跳动/...               # 面经归档
│  ├─ papers/...                # 论文归档与 manifest
│  └─ courses/...               # 课程归档
├─ reports/                     # Markdown 成品区
│  ├─ 字节跳动/LLM基础/...        # 面经报告
│  ├─ 论文精读/...               # 论文精读
│  ├─ 体系化课程/...             # 课程整理
│  └─ 00_行业洞察/...            # 面经风向标
├─ docs/                        # VitePress 站点
├─ workflow_scripts/            # 自动化脚本
│  ├─ process.py                # 面经解析主流程
│  ├─ process_paper.py          # 论文解析主流程
│  ├─ process_course.py         # 课程精编主流程
│  ├─ collect_openreview.py     # OpenReview 采集器
│  ├─ scraper_arxiv.py          # ArXiv 采集器
│  ├─ insight.py                # 面经周报
│  ├─ build_docs.py             # 站点构建
│  ├─ receiver_server.py        # Webhook 收单
│  └─ ingest_utils.py           # 文本 / 图片 / PDF 读入工具
├─ config.yaml                  # 路径与模型配置
├─ taxonomy.yaml                # 公司 / 标签 / 论文方向词表
└─ README.md
```

## 你的两类核心需求

### 1. 互联网大厂面经采集

推荐入口：

- 小红书截图直接丢进 `raw_data/`
- 小红书正文、群聊整理稿、PDF 面经集直接丢进 `raw_data/`
- 然后运行：

```bash
python workflow_scripts/process.py
python workflow_scripts/insight.py
python workflow_scripts/build_docs.py
npm run docs:build
```

当前已支持：

- 文本：`.txt` / `.md`
- 图片：`.png` / `.jpg` / `.jpeg` / `.webp` / `.bmp`
- PDF：优先提文本；若是扫描版且已安装 `PyMuPDF`，会按页转图送给多模态模型

### 2. 顶会论文采集

推荐把“采集”和“精读”拆开：

1. 先用 `collect_openreview.py` 拉官方 accepted 论文元数据
2. 再用 `process_paper.py` 做中文精读

ICLR 2026 示例：

```bash
python workflow_scripts/collect_openreview.py ^
  --invitation ICLR.cc/2026/Conference/-/Submission ^
  --accepted-venueid ICLR.cc/2026/Conference ^
  --venue-label "ICLR 2026" ^
  --min-avg-rating 7.5 ^
  --max-results 20

python workflow_scripts/process_paper.py
python workflow_scripts/build_docs.py
npm run docs:build
```

上面的逻辑会把高分 accepted 论文先写进 `raw_data_papers/`，并在 `archive/papers/_manifests/` 留一份 JSONL manifest。

## 当前结构判断

这个仓库已经不是单纯“面经题库”，而是一个 `多源资料采集 + LLM 结构化加工 + 静态站点发布` 系统。  
如果后续你继续扩源，最合理的方向不是再堆脚本，而是保持这三层稳定：

1. `采集层`：只负责把原始资料落盘，并保留来源元数据。
2. `加工层`：分类、解析、总结、打标签。
3. `发布层`：统一用 `reports/ -> docs/` 的方式做站点输出。
