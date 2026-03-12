# -*- coding: utf-8 -*-
"""
ArXiv 顶会论文每日自动爬虫 (基于 arxiv 官方 SDK 版)
自动抓取 cs.CL, cs.AI 最新论文。
"""

import os
import re
import arxiv
from datetime import datetime

# 搜索关键词，拉取大模型相关的论文
SEARCH_QUERY = 'cat:cs.CL OR cat:cs.AI OR all:"Large Language Model"'
MAX_RESULTS = 5  # 每次仅抓取最新 5 篇

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "raw_data_papers")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_arxiv_papers():
    print(f"[*] 正在通过 arxiv SDK 请求最新文献: {SEARCH_QUERY}")
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=SEARCH_QUERY,
        max_results=MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    
    saved_count = 0
    try:
        results = client.results(search)
        for paper in results:
            title = paper.title.replace('\n', ' ')
            summary = paper.summary.replace('\n', ' ')
            authors = [a.name for a in paper.authors]
            pdf_link = paper.pdf_url
            
            # 我们直接把所有抓取到的这 5 篇前沿文章存下来，让大模型去决定这篇值不值得精读
            safe_title = re.sub(r'[\\/*?:"<>|]', "", title)[:60]
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"arxiv_{timestamp}_{safe_title}.txt"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            content = f"""【来源】: ArXiv
【标题】: {title}
【作者】: {', '.join(authors)}
【链接】: {pdf_link}

【摘要】:
{summary}
"""
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
                
            print(f"[√] 成功抓取: {title[:50]}... -> {filename}")
            saved_count += 1
            
    except Exception as e:
        print(f"[-] 抓取失败: {e}")
        return
        
    print(f"\n[*] 抓取任务完成！共计获取 {saved_count} 篇最新前沿论文放入 raw_data_papers 待后台大模型精读。")

if __name__ == "__main__":
    fetch_arxiv_papers()
