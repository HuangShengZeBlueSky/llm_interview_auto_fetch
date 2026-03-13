# -*- coding: utf-8 -*-
"""
顶会论文自动解析工作流 (Papers Pipeline)
专门针对 raw_data_papers 中的文本或 PDF 提取出：论文名、核心亮点、深度分析和工业 Insight。
"""

import os
import shutil
import yaml
import json
import re
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from ingest_utils import PDF_EXTENSIONS, TEXT_EXTENSIONS, extract_text_from_pdf, read_text_file

load_dotenv()

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_taxonomy(yaml_path="taxonomy.yaml"):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_taxonomy(data, yaml_path="taxonomy.yaml"):
    with open(yaml_path, "w", encoding="utf-8", newline='\n') as f:
        yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

def parse_json_from_llm(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except:
                pass
    return {"tags": ["其他"], "new_proposed_tags": []}


def dedupe_preserve_order(items, default_value):
    seen = set()
    result = []
    for item in items or []:
        normalized = str(item).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result or [default_value]

def main():
    print("[*] 正在加载论文解析流...")
    config = load_config()
    taxonomy_path = 'taxonomy.yaml'
    taxonomy = load_taxonomy(taxonomy_path)
    
    api_key = os.environ.get("LLM_API_KEY", config['llm'].get('api_key'))
    client = OpenAI(api_key=api_key, base_url=config['llm']['base_url'], timeout=300.0)
    model_name = config['llm']['model']
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = config.get("paths", {})
    RAW_DIR = os.path.join(BASE_DIR, paths.get("raw_data_papers", "raw_data_papers"))
    REPORT_DIR = os.path.join(BASE_DIR, paths.get("reports", "reports"), "论文精读")
    ARCHIVE_DIR = os.path.join(BASE_DIR, paths.get("archive", "archive"), "papers")
    
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    
    raw_files = os.listdir(RAW_DIR)
    if not raw_files:
        print("[-] raw_data_papers 目录下没有需要处理的文献。")
        return
        
    print(f"[*] 发现 {len(raw_files)} 篇文献，开始顶会论文速读...")
    known_tags = taxonomy.get("papers_tags", ["LLM Architecture"])
    
    for filename in raw_files:
        file_path = os.path.join(RAW_DIR, filename)
        if os.path.isdir(file_path): continue
            
        print(f"\n[->] 阅读文献: {filename}")
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in TEXT_EXTENSIONS:
            content = read_text_file(file_path)
        elif file_ext in PDF_EXTENSIONS:
            content, pdf_warnings = extract_text_from_pdf(file_path)
            for warning in pdf_warnings:
                print(f"    [PDF] {warning}")
            if not content.strip():
                print("    [!] 跳过该 PDF：未提取到可用文本，请先安装 pypdf/PyMuPDF 或转为文本版 PDF。")
                continue
        else:
            print(f"    [!] 跳过格式: {file_ext}")
            continue
            
        try:
            # Stage 1: 分类与打标
            print("    [Stage 1] 正在归类研究方向...")
            sys_s1 = f"""你是一个顶会审稿人。请阅读以下论文摘要/全文。
已知的领域标签有：{known_tags}
请选出1~2个最符合该论文的标签。如无符合，可在 new_proposed_tags 提出1个新领域标签。
只能返回JSON：
{{
  "tags": ["RLHF & Reward Model"],
  "new_proposed_tags": []
}}"""
            res_s1 = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": sys_s1}, {"role": "user", "content": content}],
                temperature=0.1
            )
            meta_info = parse_json_from_llm(res_s1.choices[0].message.content)
            
            tags = dedupe_preserve_order(meta_info.get("tags", ["其他"]), "其他")
            main_tag = tags[0].replace("/", "_") if tags else "其他"
            new_tags = dedupe_preserve_order(meta_info.get("new_proposed_tags", []), "")
            new_tags = [tag for tag in new_tags if tag]
            
            taxonomy_updated = False
            for tag_candidate in tags + new_tags:
                if tag_candidate and tag_candidate not in taxonomy['papers_tags']:
                    taxonomy['papers_tags'].append(tag_candidate)
                    taxonomy_updated = True
            if taxonomy_updated:
                save_taxonomy(taxonomy, taxonomy_path)
                
            # Stage 2: 结构化解析
            print("    [Stage 2] 正在生成深度学术洞察...")
            sys_s2 = """你是一个世界顶尖的 AI 首席科学家。请为这篇论文写一份一针见血的中文解析报告。
你 MUST 采用如下 Markdown 结构（请严格按照此结构输出）：

# 📄 [在这里写出论文的中文翻译名或原名]

**作者/机构**：[提取机构和作者]  
**收录会议**：[提取会议名或写ArXiv]

## 🌟 核心亮点 (Core Highlights)
> [用1~2句通俗易懂的话，讲清楚这篇论文到底解决了什么痛点，最大突破是什么？]

## 🔬 详细方法与技术深度
[详细介绍它的网络结构、Loss函数、或者数据构造方式的创新点。如果你只是读了摘要，请合理根据你脑海中的深厚知识推演其大概的实现技术方案。]

## 💡 工业界落地 Insight
[这篇论文的技术是否适合在普通企业的业务中落地？有什么局限性？需要多少算力成本？它的 ROI 如何？这是一份极其重要的商业洞察评判。]
"""
            res_s2 = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": sys_s2}, {"role": "user", "content": content}],
                temperature=0.7
            )
            report_content = res_s2.choices[0].message.content
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            basename = os.path.splitext(filename)[0][:30]
            
            tag_report_dir = os.path.join(REPORT_DIR, main_tag)
            tag_archive_dir = os.path.join(ARCHIVE_DIR, main_tag)
            os.makedirs(tag_report_dir, exist_ok=True)
            os.makedirs(tag_archive_dir, exist_ok=True)
            
            report_path = os.path.join(tag_report_dir, f"{timestamp}_Paper_{basename}.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            
            archive_path = os.path.join(tag_archive_dir, filename)
            if os.path.exists(archive_path):
                archive_path = os.path.join(tag_archive_dir, f"{timestamp}_{filename}")
            shutil.move(file_path, archive_path)
            print(f"    - [√] 论文精读保存: {report_path}")
            
        except Exception as e:
            print(f"    [X] 失败: {str(e)}")

    print("\n[*] 论文批处理工作流执行完毕！")

if __name__ == "__main__":
    main()
