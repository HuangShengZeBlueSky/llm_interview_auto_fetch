@echo off
REM 模块：Github自动化推送脚本
REM 此脚本负责将生成的报告推送至远端仓库

chcp 65001 >nul
echo [*] 开始执行 Github 推送工作流...

cd ..\reports
if errorlevel 1 (
    echo [!] 找不到 reports 目录，请检查路径。
    pause
    exit /b
)

REM 检查是否已经初始化了 git
if not exist ".git" (
    echo [!] reports 目录尚未初始化 Git 仓库!
    echo 请先在 reports 目录下执行 git init，并连接到你的远程仓库 (git remote add origin YOUR_REPO_URL)。
    pause
    exit /b
)

echo [*] 正在添加更改...
git add .
set mytime=%time:~0,2%:%time:~3,2%:%time:~6,2%
set mytime=%mytime: =0%

echo [*] 正在提交更改...
git commit -m "Auto-update: 添加新解析的面试题报告 %date% %mytime%"

echo [*] 正在推送到远程仓库...
git push origin main
REM 如果你的默认分支是 master 可以改成 git push origin master

echo [*] 推送任务完成!
timeout /t 3
