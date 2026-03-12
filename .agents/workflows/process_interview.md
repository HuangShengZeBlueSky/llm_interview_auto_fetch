---
description: 自动批处理面试题库并更新静态知识库
---

执行大模型面试题库 V2 的完整分类处理与知识库发布流水线。

// turbo-all

1. 执行主处理程序引擎 (Two-Stage Pipeline)
系统智能识别公司和类别，按 Taxonomy 目录保存，并进行深度解析解答。
使用终端执行命令：`python workflow_scripts/process.py`

2. 重建 VitePress 静态站点索引
遍历生成的多级分类目录，自动更新站点左侧导航树。
使用终端执行命令：`python workflow_scripts/build_docs.py`

3. 部署知识库到 Web & 更新 Github 源码
部署最新的前端代码到 Github Pages，并提交所有源码包。
使用终端执行命令：`npm run docs:deploy; git add .; git commit -m "auto: 更新面试题解析和静态站点"; git push origin master`
