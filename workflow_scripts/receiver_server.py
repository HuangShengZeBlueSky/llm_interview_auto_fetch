# -*- coding: utf-8 -*-
"""
Webhook 收图服务端 (Bot Ingestion Receiver)
这是一个极其轻量的服务器。你可以让 iOS 捷径、微信机器人、或者 Telegram 机器人向它发送 HTTP 记录。
收到文件后，它会自动把文件存入 raw_data 文件夹，并在后台自动触发面试题解析工作流。
"""

import os
import shutil
import subprocess
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
import uvicorn
from pydantic import BaseModel

app = FastAPI(title="大模型面试题库 Webhook 接收端")

# 确保目录存在
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_data")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

def trigger_workflow():
    """在后台静默触发解析脚本"""
    print("[Webhook] 收到新数据，触发后台执行 process.py...")
    process_script = os.path.join(BASE_DIR, "workflow_scripts", "process.py")
    # 不阻塞当前服务器响应，丢给后台去慢慢调大模型
    subprocess.Popen(["python", process_script], cwd=BASE_DIR)

@app.post("/upload")
async def upload_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),
    text: str = Form(None)
):
    """
    接收终端发送的图片或者文本。
    你可以用 iOS 捷径或者 Postman 发送 FormData (file=图片 / text=纯文本)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_flag = False

    if file:
        # 如果收到的是图片或文件
        ext = os.path.splitext(file.filename)[1] or ".jpg"
        save_name = f"webhook_{timestamp}{ext}"
        save_path = os.path.join(RAW_DATA_DIR, save_name)
        
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"[Webhook] 已保存文件: {save_name}")
        saved_flag = True

    elif text:
        # 如果收到的是纯文本/题干
        save_name = f"webhook_{timestamp}.txt"
        save_path = os.path.join(RAW_DATA_DIR, save_name)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[Webhook] 已保存文本片段: {save_name}")
        saved_flag = True
        
    if saved_flag:
        # 数据落盘后，吩咐后台去跑大模型解析
        background_tasks.add_task(trigger_workflow)
        return {"status": "success", "msg": "资料已入库，后台正在用大模型深度解析！"}
    else:
        return {"status": "error", "msg": "未检测到有效的文件或文本！"}

if __name__ == "__main__":
    print(f"[*] Webhook 服务器启动！监听端口 8000")
    print(f"[*] POST 接口地址: http://127.0.0.0:8000/upload")
    uvicorn.run(app, host="0.0.0.0", port=8000)
