# -*- coding: utf-8 -*-
"""
模块：大模型面试题库自动化处理脚本 V2 (Two-Stage Pipeline)

【I (Input) - 输入阶段】: 
1. `config.yaml`: 读取 API 密钥、模型名称以及各个目录的相对路径。
2. `taxonomy.yaml`: 读取分类词典（公司、标签）。
3. `raw_data/`: 读取该目录下存放的所有待处理面试题。
4. `prompts/qa_template.md`: 加载用于约束大模型输出格式的提示词。

【P (Process) - 处理阶段】: 
1. 初始化：检查并自动创建缺失的文件目录。
2. 扫描文件：遍历 `raw_data/`，转为 Base64 或读取文本。
3. Stage 1 (分类提取)：调用大模型强制输出 JSON，提取题目所属的公司和知识标签。遇到新标签则自动追加到 taxonomy.yaml。
4. 动态建库：根据提取的 Company 和 Tag 生成专属分类目录。
5. Stage 2 (深度解析)：携带分类上下文，调用大模型生成万字长文解析。
6. 状态流转：处理成功的文件移动至对应的 `archive/Company/Tag/` 目录中。

【O (Output) - 输出阶段】: 
1. 将解答内容存放到对应的 `reports/Company/Tag/` 目录下。
"""

import os
import shutil
import base64
import yaml
import time
import json
import re
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件（如果存在）
load_dotenv()

# ==================== P: 阶段辅助函数 ====================

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在，请先创建！")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_taxonomy(yaml_path="taxonomy.yaml"):
    if not os.path.exists(yaml_path):
        return {"companies": ["未知"], "tags": ["其他"]}
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_taxonomy(data, yaml_path="taxonomy.yaml"):
    with open(yaml_path, "w", encoding="utf-8", newline='\n') as f:
        yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

def load_prompt(prompt_path):
    if not os.path.exists(prompt_path):
        return "你是一个资深大模型算法工程师，请详细解答以下问题。" 
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

def encode_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def init_directories(paths):
    for key, path in paths.items():
        if key not in ['prompts', 'taxonomy']:
            os.makedirs(path, exist_ok=True)

def parse_json_from_llm(text):
    """尝试从 LLM 回复中剥离 Markdown 的 JSON 代码块并解析"""
    try:
        # 如果模型返回了纯 JSON
        return json.loads(text)
    except:
        # 如果模型包在了 ```json 和 ``` 里面
        match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except:
                pass
    return {"company": "未知", "tags": ["其他"], "new_proposed_tags": []}

# ==================== 核心主流程 ====================

def main():
    print("[*] 正在加载配置和分类词库...")
    config = load_config()
    paths = config['paths']
    taxonomy_path = paths.get('taxonomy', 'taxonomy.yaml')
    taxonomy = load_taxonomy(taxonomy_path)
    
    api_key_env_name = config['llm'].get('api_key_env_var', 'LLM_API_KEY')
    api_key = os.environ.get(api_key_env_name)
    if not api_key:
       api_key = config['llm'].get('api_key')
       
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        print(f"[-] 致命错误：找不到 API 密钥！\n请设置环境变量 {api_key_env_name}！")
        return
        
    client = OpenAI(
        api_key=api_key,
        base_url=config['llm']['base_url'],
        timeout=300.0  
    )
    model_name = config['llm']['model']
    
    init_directories(paths)
    system_prompt_s2 = load_prompt(os.path.join(paths['prompts'], "qa_template.md"))
    
    raw_files = os.listdir(paths['raw_data'])
    if not raw_files:
        print("[-] raw_data 目录下没有需要处理的文件。")
        return
        
    print(f"[*] 发现 {len(raw_files)} 个待处理文件，开始执行 V2 分类问答流程...")
    
    for filename in raw_files:
        file_path = os.path.join(paths['raw_data'], filename)
        if filename.startswith('.') or os.path.isdir(file_path):
            continue
            
        print(f"\n[->] 正在处理: {filename}")
        file_ext = os.path.splitext(filename)[1].lower()
        
        # 提取文件内容格式 (文本或图片 Base64)
        input_content = None
        if file_ext in ['.txt', '.md']:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            input_content = f"原文件名：{filename}\n请解析这段文本：\n{content}"
        elif file_ext in ['.png', '.jpg', '.jpeg']:
            base64_img = encode_image_base64(file_path)
            input_content = [
                {"type": "text", "text": f"原文件名：{filename}\n请解析这张图里的面试题。"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ]
        else:
            print(f"    [!] 跳过格式: {file_ext}")
            continue

        try:
            # ---------------------------------------------------------
            # Stage 1: 意图与分类提取 (Classification)
            # ---------------------------------------------------------
            print("    [Stage 1] 正在让 AI 识别公司与知识标签...")
            system_prompt_s1 = f"""你是一个智能分类器。请阅读用户提供的面试题。
已知存在的公司有：{taxonomy['companies']}
已知存在的知识标签有：{taxonomy['tags']}

请判断这些题目属于哪家公司，并选出1到2个最贴切的标签。
如果预设标签完全不符合，你可以在 new_proposed_tags 中提出 1 个最新的精华前沿标签（如没有则留空）。
必须且只能返回合法的 JSON 对象。格式如下：
{{
  "company": "字节跳动",
  "tags": ["LLM基础", "RAG与向量检索"],
  "new_proposed_tags": []
}}
"""
            msg_s1 = [
                {"role": "system", "content": system_prompt_s1},
                {"role": "user", "content": input_content}
            ]
            # 这里即使在普通 ChatCompletion API 也可以要求返回 JSON（通过 response_format={"type": "json_object"}）
            # 注意：某些兼容 API 可能不支持 response_format，因此同时在 Prompt 里强调
            resp_s1 = client.chat.completions.create(
                model=model_name,
                messages=msg_s1,
                temperature=0.1
            )
            s1_text = resp_s1.choices[0].message.content
            meta_info = parse_json_from_llm(s1_text)
            
            company = meta_info.get("company", "未知").replace("/", "_")
            tags = meta_info.get("tags", ["其他"])
            main_tag = tags[0].replace("/", "_") if tags else "其他"
            new_tags = meta_info.get("new_proposed_tags", [])
            
            print(f"    - 识别结果：公司[{company}], 标签{tags}, 新提议标签{new_tags}")
            
            # 自动维护 Taxonomy
            taxonomy_updated = False
            if company not in taxonomy['companies'] and company != "未知":
                taxonomy['companies'].append(company)
                taxonomy_updated = True
            for nt in new_tags:
                if nt and nt not in taxonomy['tags']:
                    taxonomy['tags'].append(nt)
                    taxonomy_updated = True
            if taxonomy_updated:
                save_taxonomy(taxonomy, taxonomy_path)
                print(f"    - [√] 分类字典已自动进化，新增词汇保存到 {taxonomy_path}")

            # ---------------------------------------------------------
            # Stage 2: 深度解析 (Deep QA)
            # ---------------------------------------------------------
            print("    [Stage 2] 正在生成深度解答报告...")
            # 在 Prompt 里面悄悄补充分类信息作为上下文
            dynamic_context = f"\n\n[系统提示：本篇面试题所属公司为 {company}，主要考察知识点为：{', '.join(tags)}。请在解答时尽量贴合该公司的技术栈或该知识点的深度标准。]"
            msg_s2 = [
                {"role": "system", "content": system_prompt_s2 + dynamic_context},
                {"role": "user", "content": input_content}
            ]
            resp_s2 = client.chat.completions.create(
                model=model_name,
                messages=msg_s2,
                temperature=0.7
            )
            answer_content = resp_s2.choices[0].message.content
            
            # ---------------------------------------------------------
            # 动态建库与保存
            # ---------------------------------------------------------
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            basename = os.path.splitext(filename)[0]
            
            # 创建分类子目录
            company_report_dir = os.path.join(paths['reports'], company, main_tag)
            company_archive_dir = os.path.join(paths['archive'], company, main_tag)
            os.makedirs(company_report_dir, exist_ok=True)
            os.makedirs(company_archive_dir, exist_ok=True)
            
            report_filename = f"{timestamp}_{basename}.md"
            report_path = os.path.join(company_report_dir, report_filename)
            
            header_tags = f"**公司**：{company} | **标签**：{', '.join(tags)}"
            final_report = f"# {basename} 解析报告\n\n> {header_tags}\n> 来源文件：`{filename}`\n> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n{answer_content}"
            
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(final_report)
            print(f"    - [√] 报告已保存: {report_path}")
            
            archive_path = os.path.join(company_archive_dir, filename)
            if os.path.exists(archive_path):
                archive_path = os.path.join(company_archive_dir, f"{timestamp}_{filename}")
            shutil.move(file_path, archive_path)
            print(f"    - [√] 原件已归档: {archive_path}")
            
            time.sleep(1)
            
        except Exception as e:
            print(f"    [X] 失败: {str(e)}")

    print("\n[*] V2 分类处理工作流执行完毕！")

if __name__ == "__main__":
    main()
