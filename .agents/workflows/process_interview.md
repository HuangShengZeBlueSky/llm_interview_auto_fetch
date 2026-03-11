---
description: 自动批处理面试题库并推送Github
---

执行大模型面试题库的完整处理与推送流水线。

// turbo-all

1. 执行主处理程序引擎
系统扫描 `raw_data/` 内的新图像/文本文件，调用 LLM 进行知识拆解与解答，在 `reports/` 输出 MD 格式文档。
使用终端执行命令：`python workflow_scripts/process.py`

2. 同步推送 Github
将生成的最新题解文件 Push 至 Github 远端仓库。
使用终端执行命令：`git add .; git commit -m "auto: 更新面试题解析报告"; git push origin master`
