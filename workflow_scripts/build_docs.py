# -*- coding: utf-8 -*-
"""
生成 VitePress 静态网站配置文件及多级目录树
此脚本会在 GitHub Push 之前被 Workflow 自动调用。
"""
import os
import shutil
import json

def build():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    REPORTS_DIR = os.path.join(BASE_DIR, "reports")
    DOCS_DIR = os.path.join(BASE_DIR, "docs")
    DOCS_REPORTS_DIR = os.path.join(DOCS_DIR, "reports")
    VITEPRESS_DIR = os.path.join(DOCS_DIR, ".vitepress")
    
    os.makedirs(VITEPRESS_DIR, exist_ok=True)
    
    # 步骤1：安全地将 reports/ 同步到 docs/reports/ 以供 VitePress 编译
    if os.path.exists(DOCS_REPORTS_DIR):
        shutil.rmtree(DOCS_REPORTS_DIR)
    if os.path.exists(REPORTS_DIR):
        shutil.copytree(REPORTS_DIR, DOCS_REPORTS_DIR)
        
    # 添加一个索引文件到 docs/reports/
    with open(os.path.join(DOCS_REPORTS_DIR, "index.md"), "w", encoding="utf-8") as f:
        f.write("# 📑 最新面经汇总\n\n请从左侧边栏选择各公司或具体知识点进行浏览。")

    # 步骤2：遍历目录，生成 Sidebar 配置
    sidebar = []
    
    if os.path.exists(REPORTS_DIR):
        companies = [d for d in os.listdir(REPORTS_DIR) if os.path.isdir(os.path.join(REPORTS_DIR, d))]
        for company in companies:
            company_path = os.path.join(REPORTS_DIR, company)
            company_item = {
                "text": company.replace("_", "/"),
                "collapsed": False,
                "items": []
            }
            
            tags = [d for d in os.listdir(company_path) if os.path.isdir(os.path.join(company_path, d))]
            for tag in tags:
                tag_path = os.path.join(company_path, tag)
                tag_item = {
                    "text": tag,
                    "collapsed": True,
                    "items": []
                }
                
                md_files = [f for f in os.listdir(tag_path) if f.endswith(".md")]
                for md_file in md_files:
                    title = md_file.split("_", 2)[-1].replace(".md", "") 
                    if not title:
                        title = md_file
                    
                    # 相对链接在 VitePress 里的格式： /reports/公司/标签/文件名
                    link = f"/reports/{company}/{tag}/{md_file.replace('.md', '')}"
                    tag_item["items"].append({
                        "text": title,
                        "link": link
                    })
                
                if tag_item["items"]:
                    company_item["items"].append(tag_item)
            
            if company_item["items"]:
                sidebar.append(company_item)

    # 步骤3：生成 config.js 代码
    config_content = f"""
import {{ defineConfig }} from 'vitepress'

export default defineConfig({{
  title: "LLM 面试题库",
  description: "AI 自动提取与全量解析",
  base: "/llm_interview_auto_fetch/",
  themeConfig: {{
    nav: [
      {{ text: '🏠 首页', link: '/' }},
      {{ text: '📚 题库大全', link: '/reports/' }}
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
    config_path = os.path.join(VITEPRESS_DIR, "config.mjs")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_content.strip())
        
    print("[√] docs/.vitepress/config.mjs 已生成！路由重建完毕。")

if __name__ == "__main__":
    build()
