# 🧠 大模型面试题库 — 全自动解析 & 推送系统

> 一键搜集面试题、自动调用大模型深度解答、生成 Markdown 知识库，并同步推送至 GitHub。

## ✨ 核心功能

| 功能 | 说明 |
|------|------|
| 📸 **图片题目识别** | 截图丢进 `raw_data/`，自动 OCR + 深度解答 |
| 📝 **文本题目解析** | `.txt` / `.md` 直接读取并详细解答 |
| 📦 **标准化输出** | 每题包含：题目描述 → 前置知识 → 核心解答 → 面试追问 |
| 🚀 **一键推送** | 生成的报告自动 commit & push 到 GitHub |

## 📁 项目结构

```
大模型面试题库/
├── raw_data/              ← 【输入】你搜集的图片/文本面试题丢这里
├── reports/               ← 【输出】大模型生成的解答报告（Markdown）
├── archive/               ← 【归档】处理成功的原始文件自动移入
├── prompts/
│   └── qa_template.md     ← 提示词模板（控制解答格式）
├── workflow_scripts/
│   └── process.py         ← 核心处理引擎
├── config.yaml            ← 模型配置（模型名、API地址）
├── .env                   ← 🔒 本地密钥文件（不会上传 GitHub）
├── requirements.txt       ← Python 依赖
└── README.md              ← 本文件
```

## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/HuangShengZeBlueSky/llm_interview_auto_fetch.git
cd llm_interview_auto_fetch
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置 API 密钥
在项目根目录创建 `.env` 文件（**此文件已被 .gitignore 忽略，不会上传**）：
```
LLM_API_KEY=你的API密钥
```

### 4. 配置模型（可选）
编辑 `config.yaml`，修改模型名称或 API 地址：
```yaml
llm:
  api_key_env_var: "LLM_API_KEY"
  base_url: "https://api123.icu/v1"
  model: "claude-opus-4-6"
```

### 5. 开始使用
将面试题图片或文本丢入 `raw_data/` 目录，然后运行：
```bash
python workflow_scripts/process.py
```

## 📋 工作流程 (IPO)

```
┌─────────────┐      ┌──────────────────────┐      ┌─────────────────┐
│  I (Input)  │      │    P (Process)        │      │   O (Output)    │
│             │──────▶│                      │──────▶│                 │
│  raw_data/  │      │  Claude Opus 深度解析  │      │  reports/*.md   │
│  图片 & 文本 │      │  格式化 & 归档管理     │      │  GitHub 推送    │
└─────────────┘      └──────────────────────┘      └─────────────────┘
```

## 🔒 安全说明

- API 密钥存储在 `.env` 文件中，通过 `python-dotenv` 加载
- `.env` 已加入 `.gitignore`，**永远不会被推送到 GitHub**
- `config.yaml` 中不包含任何敏感信息，可以安全上传

## 📄 License

MIT
