# -*- coding: utf-8 -*-
"""
生成面经知识库的高阶洞察周报 (Weekly Insight)
此脚本读取最近添加到 reports/ 目录下的面经，交给大模型提取趋势和高频考点。
"""
import os
import yaml
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_recent_reports(reports_dir, max_files=10):
    """扫描所有子目录，按文件名（带时间戳）倒序拿最近的 N 个 Markdown"""
    all_files = []
    for root, _, files in os.walk(reports_dir):
        # 忽略已经生成的洞察报告
        if '00_行业洞察' in root:
            continue
        for f in files:
            if f.endswith('.md'):
                all_files.append(os.path.join(root, f))
                
    # `2026MMDD_HHMMSS_xxx.md` 这样的命名，直接按名字字符串逆序就是最新的
    all_files.sort(reverse=True)
    return all_files[:max_files]

def main():
    print("[*] 正在启动大模型洞察分析引擎...")
    config = load_config()
    
    api_key = os.environ.get(config['llm'].get('api_key_env_var', 'LLM_API_KEY'))
    client = OpenAI(
        api_key=api_key,
        base_url=config['llm']['base_url'],
        timeout=180.0
    )
    model_name = config['llm']['model']
    
    reports_dir = config['paths']['reports']
    recent_files = get_recent_reports(reports_dir, max_files=15)
    
    if not recent_files:
        print("[-] 没有找到足够的新面经来生成洞察。")
        return
        
    print(f"[*] 收集到 {len(recent_files)} 篇最近的面经，正在提取核心知识储备库...")
    
    combined_content = ""
    for file_path in recent_files:
        with open(file_path, "r", encoding="utf-8") as f:
            # 只取前 2000 个字符防超长，或者全塞进（Gemini Pro Context 很大）
            text = f.read()
            # 缩减长代码段以节省 context
            combined_content += f"\n\n--- 来源：{os.path.basename(file_path)} ---\n"
            combined_content += text[:4000] 

    system_prompt = """你是一个顶级的 AI 算法技术总监。
我将提供最近收集的 10-15 篇大厂大模型（LLM/AIGC）真实面经的片段。
请你以宏观的视野，写一篇题为《🎯 知识库大盘洞察风向标》的高水平分析周报。

要求包含以下结构：
1. **本期核心趋势**：一句话总结本周大厂最爱问的 1-2 个知识大类。
2. **高频考点红榜**：提取出出现最频繁的具体概念（比如：RoPE, PagedAttention, LoRA），给出它为什么这么火的业务背景解释。
3. **能力代差分析**：面试官目前在区分“普通候选人”和“高级候选人”时，最喜欢用的连环追问是什么？
4. **行动建议**：给系统用户（求职者）的复习优先级建议。

格式：纯粹的 Markdown 排版，使用恰当的 emoji。输出内容必须深刻、硬核。
"""

    print("[->] 正在长文本推理与风向标归纳中 (这可能需要 30-60 秒)...")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": combined_content}
            ],
            temperature=0.4
        )
        insight_content = response.choices[0].message.content
        
        # 保存结构：放在 reports/00_行业洞察/ 下面，VitePress 侧边栏会自动将其排在最上面
        insight_dir = os.path.join(reports_dir, "00_行业洞察", "最新风向标")
        os.makedirs(insight_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        export_path = os.path.join(insight_dir, f"{timestamp}_大盘洞察.md")
        
        with open(export_path, "w", encoding="utf-8") as f:
            f.write(f"# 📅 自动快照：{timestamp}\n\n{insight_content}")
            
        print(f"[√] 洞察报告已生成！保存在：{export_path}")
        
    except Exception as e:
        print(f"[X] 生成洞察失败: {str(e)}")

if __name__ == "__main__":
    main()
