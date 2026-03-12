# 🧠 大模型面试题库全自动处理系统 (V2)

> 🚀 **本项目已上线独立的静态百科网页，请访问：**
> 👉 **[LLM 面经自动知识库 (GitHub Pages)](https://HuangShengZeBlueSky.github.io/llm_interview_auto_fetch/)** 👈

这套系统实现了从“随意接收面经截图/文本”到“生成万字解析、智能分类建库、生成周报洞察、最终渲染静态网站发布”的 **全链路工作流**。

---

## ✨ 核心大版本升级 (System V2)

1. **AI 全自动分类路由**：采用 Two-Stage Pipeline。Stage 1 先用大模型智能识别面经属于“哪家公司”和“什么知识点标签”，然后根据 `taxonomy.yaml` 动态建立多级目录。
2. **知识库网站化**：引入 `VitePress`，每次处理完自动编译成带有左侧边栏、支持全站搜索的现代静态网站，并通过 GitHub Actions 全自动部署。
3. **宏观风向标洞察**：新增 `/generate_insight` 工作流，大模型自动拉取最近面经，提炼本周技术红榜和面试连环追问套路。
4. **无头收单器 Webhook**：提供 `receiver_server.py`，支持 iOS 快捷指令、微信机器人等一键推送图片，后台静默处理全流程。

## 📁 核心目录结构 (IPO 模型)

```
llm_interview_auto_fetch/
├── raw_data/                ← 【Input】外部推送或手动存放的待处理验证截图/文本
├── reports/                 ← 【Output】按 <公司>/<知识点> 分类存放的 Markdown 报告
├── archive/                 ← 【Output】处理成功的原始文件自动移入并结构化归档
├── docs/                    ← 【Output】VitePress 网站源码核心目录
├── workflow_scripts/        ← 【Process】所有的自动化脚本逻辑
│   ├── process.py           ← 两阶段分类问答引擎
│   ├── build_docs.py        ← VitePress 路由自动构建脚本
│   ├── insight.py           ← 周报洞察生成器
│   └── receiver_server.py   ← FastAPI Webhook 接收端
├── config.yaml              ← 核心配置
├── taxonomy.yaml            ← AI 分类体系词库管理记录
├── .env                     ← 🔑 你的大模型密钥（切勿泄露）
└── README.md
```

## 🚀 快速使用指南

### 方式一：终端命令运行 (本地调试)
1. 把待处理的面试题文件丢进 `raw_data/`
2. 运行 V2 引擎处理引擎：
   ```bash
   python workflow_scripts/process.py
   python workflow_scripts/build_docs.py
   npm run docs:build
   ```

### 方式二：一键全自动收单 (日常使用篇)
1. 启动本机的无人值守 Webhook 服务器：
   ```bash
   python workflow_scripts/receiver_server.py
   ```
2. 服务器将在局域网暴露出一个 `http://<本机IP>:8000/upload` 接口。
3. 在你的 iPhone 上写一个快捷指令，选择图片后 `POST` 传输到该接口。
4. **完毕**。服务端会在接收图片后自动唤起后台大模型进行分类解答、归档，你只需等待片刻即可去 GitHub 查看更新。

## 🔒 隐私与安全说明
核心 API Key 只存放在 `.env` 文件或系统环境变量中，已配置严格的 `.gitignore` 保护机制，全量开源至分支不会泄露配置。
