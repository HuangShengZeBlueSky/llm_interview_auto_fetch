# 大模型面试题库全自动处理系统

> 在线站点：<https://HuangShengZeBlueSky.github.io/llm_interview_auto_fetch/>

这个仓库现在已经不是单一的“面经解析脚本”，而是一套完整的知识库流水线：

1. 收资料：截图、文本、PDF、论文摘要、课程笔记。
2. 做结构化：分类、深度解析、自动归档。
3. 出结果：生成 `reports/`，再发布到 `docs/` 的 VitePress 站点。

## 仓库结构

```text
大模型面试题库/
├─ raw_data/                    # 面经原始资料入口
├─ raw_data_papers/             # 论文原始资料入口
├─ raw_data_courses/            # 课程原始资料入口
├─ archive/                     # 原始资料归档
├─ reports/                     # Markdown 结果区
├─ docs/                        # VitePress 站点
├─ workflow_scripts/            # 所有自动化脚本
├─ config.yaml                  # 模型与路径配置
├─ taxonomy.yaml                # 公司 / 标签 / 论文方向词表
└─ requirements.txt
```

`workflow_scripts/` 里最关键的入口：

- `process.py`：面经解析主流程，支持文本 / 图片 / PDF
- `process_paper.py`：论文精读主流程，支持文本 / PDF
- `process_course.py`：课程笔记精编主流程
- `collect_openreview.py`：按 accepted + review score 抓 OpenReview 论文
- `collect_top_conference_papers.py`：自动汇总 ICLR / NeurIPS / ICML 高分 accepted 论文并按模块生成页面
- `scraper_arxiv.py`：抓 ArXiv 最新论文
- `generate_interview_bank.py`：扫描仓库已有面经并蒸馏生成“大厂 AI 算法高频题单”
- `import_github_interview_repos.py`：统一导入 GitHub 面经仓、牛客/知乎/小红书 URL，并生成“基础知识补充 + 详细解答 + 案例模拟”题卡页
- `insight.py`：从最近面经里生成风向标周报
- `build_docs.py`：把 `reports/` 生成到 VitePress
- `receiver_server.py`：Webhook 收图收文入口

更详细的目录图和使用说明见 [docs/repo-structure.md](docs/repo-structure.md)。

## 你的当前两类需求

### 1. 互联网大厂面经采集

适合的资料来源：

- 小红书截图
- 小红书正文复制后的文本
- 面经 PDF 合集
- 群聊整理稿、Word 导出的文本

推荐流程：

```bash
python workflow_scripts/process.py
python workflow_scripts/insight.py
python workflow_scripts/build_docs.py
npm run docs:build
```

默认把面经原始资料放进 `raw_data/`。  
处理完成后会进入：

- `archive/<公司>/<标签>/`
- `reports/<公司>/<标签>/`

如果你想直接生成“可刷题”的高频题单，而不是继续堆单篇面经，可以额外运行：

```bash
python workflow_scripts/generate_interview_bank.py
python workflow_scripts/build_docs.py
npm run docs:build
```

如果你想把公开题库和真实面经一口气变成网页资料卡，可以运行：

```bash
python workflow_scripts/import_github_interview_repos.py
python workflow_scripts/build_docs.py
npm run docs:build
```

外部真实面经 URL 统一维护在 `interview_source_urls.yaml`，同一条导入脚本已经支持：

- `nowcoder`
- `zhihu`
- `xiaohongshu`

### 2. 顶会论文采集

推荐做法是把“采集”和“精读”分成两步：

1. 先从官方源抓论文元数据
2. 再把结果交给 LLM 生成中文精读

ICLR 2026 accepted 高分论文示例：

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

这会把命中的论文先写进 `raw_data_papers/`，并在 `archive/papers/_manifests/` 保留 manifest。

如果你想跳过逐篇精读，先直接得到“顶会高分论文模块汇总”，可以运行：

```bash
python workflow_scripts/collect_top_conference_papers.py
python workflow_scripts/build_docs.py
npm run docs:build
```

## 安装

```bash
python -m pip install -r requirements.txt
npm install
```

如果你要处理扫描版 PDF，建议安装 `PyMuPDF`；如果只是文本版 PDF，`pypdf` 就够用。

## 快速开始

### 本地批处理

```bash
python workflow_scripts/process.py
python workflow_scripts/process_paper.py
python workflow_scripts/process_course.py
python workflow_scripts/build_docs.py
npm run docs:build
```

### Webhook 收单

```bash
python workflow_scripts/receiver_server.py
```

服务会暴露 `http://<你的IP>:8000/upload`，支持表单上传图片或文本。

## 说明

`taxonomy.yaml` 负责维护三套词表：

- `interviews_companies`
- `interviews_tags`
- `papers_tags`
- `courses_tags`

面经主流程现在会自动兼容旧版 `companies/tags` 写法，并在运行时修正为新版词表结构。
