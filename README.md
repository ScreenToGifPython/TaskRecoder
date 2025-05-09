# TaskRecoder
NewBeePlus任务管理系统

```                                                                                                                   
                              8888888 8888888888   .8.            d888888o.   8 8888     ,88'                         
                                    8 8888        .888.         .`8888:' `88. 8 8888    ,88'                          
                                    8 8888       :88888.        8.`8888.   Y8 8 8888   ,88'                           
                                    8 8888      . `88888.       `8.`8888.     8 8888  ,88'                            
                                    8 8888     .8. `88888.       `8.`8888.    8 8888 ,88'                             
                                    8 8888    .8`8. `88888.       `8.`8888.   8 8888 88'                              
                                    8 8888   .8' `8. `88888.       `8.`8888.  8 888888<                               
                                    8 8888  .8'   `8. `88888.  8b   `8.`8888. 8 8888 `Y8.                             
                                    8 8888 .888888888. `88888. `8b.  ;8.`8888 8 8888   `Y8.                           
                                    8 8888.8'       `8. `88888. `Y8888P ,88P' 8 8888     `Y8.                         
                                                                                                                      
8 888888888o.   8 8888888888       ,o888888o.        ,o888888o.     8 888888888o.      8 8888888888   8 888888888o.   
8 8888    `88.  8 8888            8888     `88.   . 8888     `88.   8 8888    `^888.   8 8888         8 8888    `88.  
8 8888     `88  8 8888         ,8 8888       `8. ,8 8888       `8b  8 8888        `88. 8 8888         8 8888     `88  
8 8888     ,88  8 8888         88 8888           88 8888        `8b 8 8888         `88 8 8888         8 8888     ,88  
8 8888.   ,88'  8 888888888888 88 8888           88 8888         88 8 8888          88 8 888888888888 8 8888.   ,88'  
8 888888888P'   8 8888         88 8888           88 8888         88 8 8888          88 8 8888         8 888888888P'   
8 8888`8b       8 8888         88 8888           88 8888        ,8P 8 8888         ,88 8 8888         8 8888`8b       
8 8888 `8b.     8 8888         `8 8888       .8' `8 8888       ,8P  8 8888        ,88' 8 8888         8 8888 `8b.     
8 8888   `8b.   8 8888            8888     ,88'   ` 8888     ,88'   8 8888    ,o88P'   8 8888         8 8888   `8b.   
8 8888     `88. 8 888888888888     `8888888P'        `8888888P'     8 888888888P'      8 888888888888 8 8888     `88. 
```


# 工作任务统计与周报生成工具

本工具是一个基于 PySide6 构建的图形化任务管理应用，支持：

- 任务的创建、执行与日志记录
- 任务耗时图表（饼图 / 柱状图）统计
- DeepSeek 大模型自动生成本周周报（支持自定义时间范围）
- 多项任务类型筛选统计图支持（如只看“开发 + 文档”）
- 本产品的idea和强大初代代码来自最强开发kevinCJM老师👏👏zyy只做了一点锦上添花的工作
## 📦 安装依赖

请使用 Python 3.8+ 环境，推荐使用 virtualenv 或 conda。

```bash
pip install -r requirements.txt
```

## 🚀 启动方式

```bash
python main.py
```

## 🧠 使用说明

- 「任务管理」页：添加任务、修改任务状态
- 「执行管理」页：启动任务、结束任务、记录执行信息
- 「执行明细」页：查看任务执行日志
- 「工作统计」页：
  - 按任务大类/细类/名称统计耗时
  - 可筛选分类项
  - 可一键生成本周周报（调用 DeepSeek API，需自备 API Key）

## 🔐 DeepSeek API Key 填写

前往 https://platform.deepseek.com 获取你的 API Key，并填写在界面下方文本框中。
或者其他大模型api均可～

## 📊 依赖包列表

详见 `requirements.txt`，主要依赖如下：

PySide6
httpx
matplotlib
openai
pandas
pyarrow

## 📁 数据存储

任务数据与日志将以 Parquet 格式保存在本地：
- `tasks.parquet`
- `task_logs.parquet`



