# 课程板块

这里按课程名聚合内容，适合持续搬运公开课、字幕稿和课程讲义。

## CS224N (NLP基础) (1)
- [Note_CS224N_Lecture1_Notes](/reports/体系化课程/CS224N-(NLP基础)/20260312_165246_Note_CS224N_Lecture1_Notes) | 2026-03-12 16:52:46

## 课程迁移接口

你现在搬课程可以直接复用现有流水线，不需要额外改数据库：

1. 把原始课程材料整理成 UTF-8 编码的 `.txt` 或 `.md` 文件，放到 `raw_data_courses/`。
2. 文件名建议用 `课程名_章节名.txt`，例如 `CS231N_Lecture01_Intro.txt`。
3. 如果想让分类更稳定，先把课程名补进根目录的 `taxonomy.yaml` -> `courses_tags`。
4. 运行 `python workflow_scripts/process_course.py`，课程内容会被整理到 `reports/体系化课程/<课程名>/`。
5. 再运行 `python workflow_scripts/build_docs.py` 和 `npm run docs:build`，站点会自动更新。

如果你的来源是 YouTube/B 站视频，建议先做一层转写，把字幕导出成文本后再进这条通道。
