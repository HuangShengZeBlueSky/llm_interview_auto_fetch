# -*- coding: utf-8 -*-
"""
体系化公开课与硬核笔记工作流 (Courses Pipeline)
针对 raw_data_courses 中的无格式长文本（如 YouTube 原始字幕、散乱笔记），
使用大模型进行“教材级精编”，去掉口语化废话，生成结构化的 Markdown 教科书章节。
"""

import os
import shutil
import yaml
import json
import re
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

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
    return {"course_name": "杂谈笔记", "tags": ["其他"], "new_proposed_tags": []}

def main():
    print("[*] 正在加载课程笔记提炼流...")
    config = load_config()
    taxonomy_path = 'taxonomy.yaml'
    taxonomy = load_taxonomy(taxonomy_path)
    
    api_key = os.environ.get("LLM_API_KEY", config['llm'].get('api_key'))
    client = OpenAI(api_key=api_key, base_url=config['llm']['base_url'], timeout=300.0)
    model_name = config['llm']['model']
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DIR = os.path.join(BASE_DIR, "raw_data_courses")
    REPORT_DIR = os.path.join(BASE_DIR, "reports", "体系化课程")
    ARCHIVE_DIR = os.path.join(BASE_DIR, "archive", "courses")
    
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    
    raw_files = os.listdir(RAW_DIR)
    if not raw_files:
        print("[-] raw_data_courses 目录下没有需要处理的字幕/笔记。")
        return
        
    print(f"[*] 发现 {len(raw_files)} 个笔记文件，开始大模型教材级精编...")
    known_tags = taxonomy.get("courses_tags", ["默认"])
    
    for filename in raw_files:
        file_path = os.path.join(RAW_DIR, filename)
        if os.path.isdir(file_path): continue
            
        print(f"\n[->] 精编片段: {filename}")
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        try:
            # Stage 1: 分类与归属
            print("    [Stage 1] 正在识别所属课程章节...")
            sys_s1 = f"""你是一个智能助教。请阅读以下原始文本片段（可能含有大量杂乱口语）。
推断这最可能是哪门课或者哪个领域的笔记。
已知的领域/课程有：{known_tags}
请选出一个最符合的课程标签。如果在列表外，可在 new_proposed_tags 提出1个新的课程名（如 CS224N）。
只能返回JSON：
{{
  "tags": ["CS25 (大模型前沿)"],
  "new_proposed_tags": []
}}"""
            res_s1 = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": sys_s1}, {"role": "user", "content": content}],
                temperature=0.1
            )
            meta_info = parse_json_from_llm(res_s1.choices[0].message.content)
            
            tags = meta_info.get("tags", ["其他"])
            main_tag = tags[0].replace("/", "_") if tags else "其他"
            new_tags = meta_info.get("new_proposed_tags", [])
            
            taxonomy_updated = False
            for nt in new_tags:
                if nt and nt not in taxonomy['courses_tags']:
                    taxonomy['courses_tags'].append(nt)
                    taxonomy_updated = True
            if taxonomy_updated:
                save_taxonomy(taxonomy, taxonomy_path)
                
            # Stage 2: 结构化解析
            print("    [Stage 2] 正在进行教材级重写(去除废话/结构化)...")
            sys_s2 = """你是一个世界顶尖的大学教授。你现在需要将一段非常散乱、甚至是有大量错别字和废话的【演讲字幕/手写笔记】重新编排为一篇极其精美的【教科书级 Markdown 章节】。
你 MUST 按照以下要求处理：
1. **彻底去除口语化词汇**：不要出现“嗯、啊、我觉得、各位同学”等。把它提炼成高度严谨的书面教案。
2. **逻辑分层**：用清晰的 H2 (##) 和 H3 (###) 标题结构化内容，如果能总结成表格，就用 Markdown 表格。
3. **补充背景说明**：如果演讲人提到了某个公式或者某个前沿名词但是没解释清楚，请你运用强大的大模型知识储备，在这个词旁边用 `> [!NOTE]` 的语法添加补充说明！
4. **统一为中文**：无论输入是全英文讲义还是中英混杂，最终产出的长文必须是信达雅的中文。

请直接输出这篇精美重写的 Markdown 笔记本身，无需带有开场白：
"""
            res_s2 = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": sys_s2}, {"role": "user", "content": f"输入的原笔记素材：\n{content}"}],
                temperature=0.6
            )
            report_content = res_s2.choices[0].message.content
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            basename = os.path.splitext(filename)[0][:30]
            
            tag_report_dir = os.path.join(REPORT_DIR, main_tag)
            tag_archive_dir = os.path.join(ARCHIVE_DIR, main_tag)
            os.makedirs(tag_report_dir, exist_ok=True)
            os.makedirs(tag_archive_dir, exist_ok=True)
            
            report_path = os.path.join(tag_report_dir, f"{timestamp}_Note_{basename}.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
                
            archive_path = os.path.join(tag_archive_dir, filename)
            shutil.move(file_path, archive_path)
            print(f"    - [√] 教材级笔记已保存: {report_path}")
            
        except Exception as e:
            print(f"    [X] 失败: {str(e)}")

    print("\n[*] 公开课笔记批处理工作流执行完毕！")

if __name__ == "__main__":
    main()
