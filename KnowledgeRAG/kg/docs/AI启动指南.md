# 知识图谱 AI Agent 启动指南

## 📖 概述

本文档介绍如何启动知识图谱 AI Agent 系统。系统提供了多种启动方式，适用于不同的使用场景。

---

## 🔧 前置准备

### 1. 确保 Neo4j 服务已启动

```powershell
# 检查 Neo4j 是否运行
# 访问 http://localhost:7474/ 或在命令行检查端口
netstat -ano | findstr :8687
```

如果未启动，请参考 [`Neo4j启动指南.md`](./Neo4j启动指南.md) 启动 Neo4j。

### 2. 激活 Python 环境

```powershell
# 激活 conda 环境
conda activate py310

# 进入项目目录
cd E:\1代码\系统\STKGRAG\kg
```

### 3. 安装依赖（如未安装）

```powershell
pip install neo4j langchain-community dashscope
```

---

## 🚀 启动方式

### 方式一：连接测试（最简单）

**用途**：快速验证 Neo4j 连接是否正常

**命令**：
```powershell
python base.py
```

**功能**：
- ✅ 测试 Neo4j 连接
- ✅ 查询所有节点（前20个）
- ✅ 查询所有关系（前20个）

**输出示例**：
```
✅ 成功连接到 Neo4j!

📘 查询所有节点
节点ID=..., 标签=['实体'], 属性={...}

🔗 查询所有关系
关系ID=..., 类型=包含实体, ... → ...
```

---

### 方式二：基础 Agent 示例（演示模式）

**用途**：运行预设的查询示例，了解 Agent 功能

**命令**：
```powershell
python kg_agent.py
```

**功能**：
- ✅ 自动运行 6 个预设查询示例
- ✅ 展示 Agent 的各种查询能力
- ✅ 无需 LLM，直接使用工具查询

**示例内容**：
1. 获取知识图谱概览
2. 查询所有实体
3. 查询特定实体（如"植被"）的详细信息
4. 查询所有指数
5. 查询状态转换关系
6. 查询时相信息

**适用场景**：
- 学习 Agent 使用方法
- 验证知识图谱数据
- 测试查询功能

---

### 方式三：智能对话 Agent（推荐）

**用途**：与知识图谱进行自然语言对话

**命令**：
```powershell
python langchain_agent.py
```

**功能**：
- ✅ 自然语言问答
- ✅ 流式输出（实时显示回答）
- ✅ 智能意图识别
- ✅ 多轮对话支持

**启动流程**：
1. 运行命令后，会提示输入 DashScope API Key
2. 输入您的 API Key（或设置环境变量 `DASHSCOPE_API_KEY`）
3. 进入对话模式

**使用示例**：
```
👤 您: 有哪些指数可以使用？
🤖 Agent: 根据知识图谱，有以下监测指数可以使用：
1. NDVI - 植被覆盖度
2. NDWI - 水体或湿度特征
3. NBR - 扰动及火烧强度
...

👤 您: 植被用什么指数监测？
🤖 Agent: 植被实体使用以下监测指数：
- NDVI（植被覆盖度）
- NDWI（水体或湿度特征）
- NBR（扰动及火烧强度）
```

**退出方式**：
- 输入 `exit`、`quit` 或 `退出`
- 按 `Ctrl+C`

**环境变量设置（可选）**：
```powershell
# Windows PowerShell
$env:DASHSCOPE_API_KEY = "your_api_key_here"

# 或在系统环境变量中设置
```

---

## 📝 统一启动脚本

为了方便使用，可以创建一个统一的启动脚本：

### 创建 `start.py`

```python
"""
知识图谱 AI Agent 统一启动脚本
"""
import sys

def show_menu():
    """显示启动菜单"""
    print("=" * 60)
    print("🤖 知识图谱 AI Agent 启动菜单")
    print("=" * 60)
    print("1. 连接测试（base.py）")
    print("2. 基础 Agent 示例（kg_agent.py）")
    print("3. 智能对话 Agent（langchain_agent.py）")
    print("0. 退出")
    print("=" * 60)

def main():
    """主函数"""
    while True:
        show_menu()
        choice = input("\n请选择启动方式 (0-3): ").strip()
        
        if choice == "0":
            print("👋 再见！")
            break
        elif choice == "1":
            print("\n正在启动连接测试...")
            import base
        elif choice == "2":
            print("\n正在启动基础 Agent 示例...")
            import kg_agent
        elif choice == "3":
            print("\n正在启动智能对话 Agent...")
            import langchain_agent
        else:
            print("❌ 无效选择，请重新输入")
        
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()
```

**使用方式**：
```powershell
python start.py
```

---

## 🎯 使用场景推荐

| 场景 | 推荐方式 | 说明 |
|------|---------|------|
| **首次使用** | `base.py` | 验证环境配置是否正确 |
| **学习功能** | `kg_agent.py` | 查看预设示例，了解功能 |
| **日常使用** | `langchain_agent.py` | 自然语言对话，最方便 |
| **调试问题** | `base.py` | 快速检查连接和数据 |

---

## ⚙️ 配置说明

### Neo4j 连接配置

所有脚本使用的配置（在代码中定义）：

```python
URI = "bolt://127.0.0.1:8687"  # Neo4j 连接地址
USER = "neo4j"                  # 用户名
PASSWORD = "qianxi147A"         # 密码
```

如需修改，请更新以下文件：
- `base.py`
- `kg_agent.py`（在 `example_usage()` 函数中）
- `langchain_agent.py`（在 `create_kg_agent()` 函数中）

### DashScope API Key 配置

**方式一：环境变量（推荐）**
```powershell
# Windows PowerShell
$env:DASHSCOPE_API_KEY = "your_api_key"

# Windows CMD
set DASHSCOPE_API_KEY=your_api_key
```

**方式二：运行时输入**
- 运行 `langchain_agent.py` 时会提示输入

**方式三：代码中设置（不推荐）**
- 在 `langchain_agent.py` 中直接设置（安全性较低）

---

## 🐛 常见问题

### 问题 1：连接失败

**错误信息**：`❌ Neo4j 连接失败`

**解决方案**：
1. 检查 Neo4j 服务是否运行
2. 确认端口配置正确（应为 8687）
3. 验证用户名和密码

### 问题 2：API Key 错误

**错误信息**：`Authentication failed` 或 `Invalid API Key`

**解决方案**：
1. 检查 DashScope API Key 是否正确
2. 确认 API Key 是否有效（未过期）
3. 检查网络连接是否正常

### 问题 3：模块导入错误

**错误信息**：`ModuleNotFoundError`

**解决方案**：
```powershell
# 安装缺失的依赖
pip install neo4j langchain-community dashscope
```

### 问题 4：知识图谱为空

**错误信息**：查询结果为空

**解决方案**：
1. 确认已执行 `创建.cypher` 脚本初始化知识图谱
2. 检查 Neo4j Browser 中是否有数据
3. 运行 `base.py` 验证数据是否存在

---

## 📊 快速启动检查清单

- [ ] ✅ Neo4j 服务已启动（端口 8687）
- [ ] ✅ Python 环境已激活（py310）
- [ ] ✅ 依赖包已安装
- [ ] ✅ 知识图谱已初始化（执行了创建脚本）
- [ ] ✅ DashScope API Key 已配置（如使用对话 Agent）

---

## 🔗 相关文档

- [`Neo4j启动指南.md`](./Neo4j启动指南.md) - Neo4j 数据库启动指南
- [`使用指南.md`](./使用指南.md) - Agent 详细使用说明
- [`关系.md`](./关系.md) - 知识图谱结构说明
- [`创建.cypher`](./创建.cypher) - 知识图谱创建脚本

---

## 💡 使用技巧

### 1. 快速测试连接

```powershell
# 一键测试
python base.py
```

### 2. 查看所有功能示例

```powershell
# 运行所有预设示例
python kg_agent.py
```

### 3. 交互式对话

```powershell
# 启动对话模式
python langchain_agent.py

# 常用问题示例：
# - "有哪些指数可以使用？"
# - "植被用什么指数监测？"
# - "滑坡恢复的各个阶段是什么？"
# - "查询所有实体"
```

### 4. 自定义查询

在代码中使用 Agent：

```python
from neo4j_tools import Neo4jTools
from kg_agent import KnowledgeGraphAgent, QueryTemplates

# 创建 Agent
with Neo4jTools("bolt://127.0.0.1:8687", "neo4j", "qianxi147A") as tools:
    agent = KnowledgeGraphAgent(tools)
    
    # 自定义查询
    result = agent.query(
        "查询所有实体",
        tool_calls=QueryTemplates.get_all_entities()
    )
    print(agent.format_response(result))
```

---

*最后更新：2025年*

