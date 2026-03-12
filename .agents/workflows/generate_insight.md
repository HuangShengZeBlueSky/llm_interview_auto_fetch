---
description: 自动总结最近知识库脉络与洞察
---

一键扫描最近更新的面经和文档，自动生成高阶版《大厂面试风向标》分析。

// turbo-all

1. 执行 AI 洞察分析脚本
提取并分析最新资料的核心考点。
使用终端执行命令：`python workflow_scripts/insight.py`

2. 重建 VitePress 导航树以包含此总结
使用终端执行命令：`python workflow_scripts/build_docs.py`

3. 同步至 Github
推送 main 分支后，云端会自动更新知识库网站！
使用终端执行命令：`git add docs/insights/; git commit -m "auto: 生成新的周报总结"; git push origin main`
