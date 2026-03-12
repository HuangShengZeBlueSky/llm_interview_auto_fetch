# -*- coding: utf-8 -*-
"""
模块：大模型面试题库自动化处理脚本

【I (Input) - 输入阶段】: 
1. `config.yaml`: 读取 API 密钥、模型名称以及各个目录的相对路径。
2. `raw_data/`: 读取该目录下存放的所有待处理面试题（图片 `.png`, `.jpg` 或 文本 `.txt`, `.md`）。
3. `prompts/qa_template.md`: 加载用于约束大模型输出格式的提示词。

【P (Process) - 处理阶段】: 
1. 初始化：检查并自动创建缺失的文件目录（raw_data, archive, reports）。
2. 扫描文件：遍历 `raw_data/`，判断文件类型。
3. 数据预处理：
   - 文本格式：直接读取内容为字符串。
   - 图片格式：读取字节并转为 Base64 编码，适配 VLM（视觉语言模型）接口。
4. 模型调用：拼装 Prompt 模板与预处理后的数据，通过 OpenAI SDK 标准接口调用目标大模型进行解答。
5. 状态流转：将处理成功的文件移动至 `archive/` 目录中，作为历史备份；失败则保留并记录。

【O (Output) - 输出阶段】: 
1. 将大模型返回的解答内容格式化为 Markdown。
2. 以 "YYYYMMDD_HHMMSS_原文件名.md" 的格式，保存到 `reports/` 目录下。
"""

import os
import shutil
import base64
import yaml
import time
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件（如果存在）
load_dotenv()

# ==================== P: 阶段辅助函数 ====================

def load_config(config_path="config.yaml"):
    """【Input】：加载项目核心配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件 {config_path} 不存在，请先创建！")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_prompt(prompt_path):
    """【Input】：加载系统提示词模板"""
    if not os.path.exists(prompt_path):
        return "你是一个资深大模型算法工程师，请详细解答以下问题。" # 默认兜底
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

def encode_image_base64(image_path):
    """【Process】：将图片转为 base64 格式，供大模型识别"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def init_directories(paths):
    """【Process】：根据配置文件，确保目录存在"""
    for key, path in paths.items():
        if key != 'prompts': # prompts 不需要在此创建，如果没有配置默认不要紧
            os.makedirs(path, exist_ok=True)

# ==================== 核心主流程 ====================

def main():
    # 1. 【Input层】获取配置和初始化客户端
    print("[*] 正在加载配置...")
    config = load_config()
    
    # 【安全更新】：优先从环境变量获取 API Key，如果没有则报错提醒
    api_key_env_name = config['llm'].get('api_key_env_var', 'LLM_API_KEY')
    api_key = os.environ.get(api_key_env_name)
    
    # 兼容老版直接写在 config.yaml 里的情况（不推荐）
    if not api_key:
       api_key = config['llm'].get('api_key')
       
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        print(f"[-] 致命错误：找不到 API 密钥！\n请设置环境变量 {api_key_env_name}，或在 config.yaml 中配置。为防止 Github 泄露，强烈建议使用环境变量！")
        return
        
    client = OpenAI(
        api_key=api_key,
        base_url=config['llm']['base_url'],
        timeout=300.0  # 5分钟超时，避免大文件处理时 504 网关超时
    )
    model_name = config['llm']['model']
    paths = config['paths']
    
    # 2. 目录初始化
    init_directories(paths)
    system_prompt = load_prompt(os.path.join(paths['prompts'], "qa_template.md"))
    
    # 3. 【Process层】扫描任务队列
    raw_files = os.listdir(paths['raw_data'])
    if not raw_files:
        print("[-] raw_data 目录下没有需要处理的文件。")
        return
        
    print(f"[*] 发现 {len(raw_files)} 个待处理文件，开始执行流程...")
    
    for filename in raw_files:
        file_path = os.path.join(paths['raw_data'], filename)
        
        # 忽略隐藏文件或文件夹
        if filename.startswith('.') or os.path.isdir(file_path):
            continue
            
        print(f"\n[->] 正在处理文件: {filename}")
        file_ext = os.path.splitext(filename)[1].lower()
        
        # 组装 API 消息体
        messages = [{"role": "system", "content": system_prompt}]
        
        try:
            if file_ext in ['.txt', '.md']:
                # 解析文本
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                messages.append({
                    "role": "user",
                    "content": f"请解答以下面试题：\n{content}"
                })
                
            elif file_ext in ['.png', '.jpg', '.jpeg']:
                # 解析图片
                base64_img = encode_image_base64(file_path)
                # 适配支持 Vision 的 OpenAI 格式
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请提取图片中的面试题，并给出详细解答。"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                        }
                    ]
                })
            else:
                print(f"[!] 跳过不支持的文件格式: {filename}")
                continue
                
            # 4. 【Process层】调用大模型 API
            print("    - 正在等待大模型生成解答...")
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7
            )
            answer_content = response.choices[0].message.content
            
            # 5. 【Output层】输出报告文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            basename = os.path.splitext(filename)[0]
            report_filename = f"{timestamp}_{basename}.md"
            report_path = os.path.join(paths['reports'], report_filename)
            
            # 追加原始题目的出处标识，方便溯源
            final_report = f"# {basename} 解析报告\n\n> 来源文件：`{filename}`\n> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n{answer_content}"
            
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(final_report)
            print(f"    - [√] 报告已生成: {report_path}")
            
            # 6. 【Process层】状态流转：处理成功，归档文件
            archive_path = os.path.join(paths['archive'], filename)
            # 如果 archive 里已经有同名文件，为了防止覆盖，加上时间戳
            if os.path.exists(archive_path):
                archive_path = os.path.join(paths['archive'], f"{timestamp}_{filename}")
            shutil.move(file_path, archive_path)
            print(f"    - [√] 文件已归档至: {archive_path}")
            
            # 避免触发 API 频率限制，短暂休眠
            time.sleep(1)
            
        except Exception as e:
            print(f"    - [X] 处理文件 {filename} 时发生错误: {str(e)}")

    print("\n[*] 本次工作流执行完毕！")

if __name__ == "__main__":
    main()
