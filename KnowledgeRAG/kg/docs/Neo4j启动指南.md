# Neo4j 从零启动指南

## 📖 概述

本文档将指导您从零开始安装、配置和启动 Neo4j 图数据库服务，并完成知识图谱的初始化。

---

## 📋 前置要求

- Windows 10/11 操作系统
- Java 11 或更高版本（Neo4j 需要 Java 运行环境）
- 管理员权限（用于安装和配置服务）

---

## 🔧 第一步：安装 Neo4j

### 方式一：使用 Neo4j Desktop（推荐）

1. **下载 Neo4j Desktop**
   - 访问官网：https://neo4j.com/download/
   - 下载 Neo4j Desktop for Windows
   - 安装并启动 Neo4j Desktop

2. **创建数据库项目**
   - 打开 Neo4j Desktop
   - 点击 "New Project" 创建新项目
   - 点击 "Add Database" → "Create a Local Database"
   - 设置数据库名称（如：`knowledge_graph`）
   - 设置密码：`qianxi147A`（与项目配置一致）

3. **配置端口**
   - 点击数据库设置（Settings）
   - 找到 "Bolt Port" 配置项
   - 修改为：`8687`（与项目配置一致）
   - 保存设置

4. **启动数据库**
   - 点击数据库卡片上的 "Start" 按钮
   - 等待数据库启动完成（状态变为绿色）

### 方式二：使用 Neo4j Community Edition（命令行）

1. **下载 Neo4j Community Edition**
   - 访问：https://neo4j.com/download-center/#community
   - 下载 Windows 版本（ZIP 压缩包）

2. **解压安装**
   ```powershell
   # 解压到指定目录，例如：C:\neo4j
   ```

3. **配置端口**
   - 打开 `conf/neo4j.conf` 文件
   - 找到 `#dbms.connector.bolt.listen_address=:7687`
   - 修改为：`dbms.connector.bolt.listen_address=:8687`
   - 保存文件

4. **设置密码**
   - 打开 `conf/neo4j.conf` 文件
   - 找到 `#dbms.security.auth_enabled=true`
   - 确保启用认证（默认已启用）

5. **启动 Neo4j**
   ```powershell
   # 方式一：进入 bin 目录后执行（推荐）
   cd D:\neo4j1\neo4j-community-5.26.0\bin
   .\neo4j.bat console

   # 方式二：在根目录使用相对路径
   cd D:\neo4j1\neo4j-community-5.26.0
   .\bin\neo4j.bat console

   # 安装为 Windows 服务（后台运行）
   cd D:\neo4j1\neo4j-community-5.26.0\bin
   .\neo4j.bat install-service
   .\neo4j.bat start

   # 停止 Neo4j
   .\neo4j.bat stop

   # 查看状态
   .\neo4j.bat status
   ```

   **⚠️ 重要提示**：在 PowerShell 中执行脚本必须使用 `.\` 前缀！

---

## 🔐 第二步：设置初始密码

### 首次启动 Neo4j Desktop

1. 启动数据库后，点击 "Open" 打开 Neo4j Browser
2. 首次登录会要求设置密码
3. 设置密码为：`qianxi147A`
4. 确认密码设置完成

### 使用命令行设置密码

```powershell
# 使用 cypher-shell 连接数据库
cd C:\neo4j\bin
.\cypher-shell.bat -a bolt://localhost:8687 -u neo4j -p neo4j

# 在 cypher-shell 中执行
ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO 'qianxi147A';
```

---

## ✅ 第三步：验证连接

### 方式一：使用 Neo4j Browser（图形界面）

1. 在 Neo4j Desktop 中点击数据库的 "Open" 按钮
2. 打开 Neo4j Browser（通常在浏览器中打开）
3. 输入连接信息：
   - **连接地址**：`bolt://localhost:8687`
   - **用户名**：`neo4j`
   - **密码**：`qianxi147A`
4. 点击 "Connect" 连接
5. 执行测试查询：
   ```cypher
   RETURN "Hello Neo4j!" AS greeting;
   ```

### 方式二：使用 Python 脚本验证

运行项目中的测试脚本：

```powershell
# 激活 conda 环境
conda activate py310

# 进入项目目录
cd E:\1代码\系统\STKGRAG\kg

# 运行连接测试
python base.py
```

如果看到 `✅ 成功连接到 Neo4j!` 说明连接成功。

---

## 📊 第四步：初始化知识图谱

### 使用 Neo4j Browser 执行脚本

1. **打开 Neo4j Browser**
   - 在 Neo4j Desktop 中点击数据库的 "Open" 按钮

2. **执行创建脚本**
   - 打开项目文件：`kg/docs/创建.cypher`
   - 复制全部内容
   - 粘贴到 Neo4j Browser 的查询框中
   - 点击 "Run" 执行
   - 等待脚本执行完成（可能需要几秒钟）

3. **验证数据创建**
   ```cypher
   // 查看节点数量
   MATCH (n) RETURN count(n) AS 节点数;
   
   // 查看关系数量
   MATCH ()-[r]->() RETURN count(r) AS 关系数;
   
   // 查看所有实体
   MATCH (e:实体) RETURN e.名称 AS 实体名称;
   ```

### 使用命令行执行脚本

```powershell
# 进入 Neo4j bin 目录
cd C:\neo4j\bin

# 使用 cypher-shell 执行脚本
.\cypher-shell.bat -a bolt://localhost:8687 -u neo4j -p qianxi147A -f "E:\1代码\系统\STKGRAG\kg\docs\创建.cypher"
```

---

## 🎯 第五步：验证知识图谱

### 执行查询脚本

1. **在 Neo4j Browser 中执行**
   - 打开 `kg/docs/查询.cypher`
   - 复制内容到 Neo4j Browser
   - 执行查询，查看图谱结构

2. **使用 Python Agent 查询**
   ```powershell
   # 运行 Agent 示例
   python kg_agent.py
   ```

### 常用验证查询

```cypher
// 查看所有节点类型
MATCH (n)
RETURN labels(n) AS 节点类型, count(n) AS 数量
ORDER BY 数量 DESC;

// 查看所有关系类型
MATCH ()-[r]->()
RETURN type(r) AS 关系类型, count(r) AS 数量
ORDER BY 数量 DESC;

// 查看实体与指数的关系
MATCH (e:实体)-[:监测指标]->(i:指数)
RETURN e.名称 AS 实体, collect(i.名称) AS 监测指数;
```

---

## 🛠️ 常见问题排查

### 问题 1：连接失败

**错误信息**：`无法连接到 Neo4j`

**解决方案**：
1. 检查 Neo4j 服务是否正在运行
   ```powershell
   # 检查端口是否被占用
   netstat -ano | findstr :8687
   ```
2. 确认端口配置正确（应为 8687）
3. 检查防火墙设置
4. 确认密码正确

### 问题 2：端口冲突

**错误信息**：`Address already in use`

**解决方案**：
1. 检查是否有其他 Neo4j 实例在运行
2. 修改端口配置（在 `neo4j.conf` 中修改）
3. 同时修改项目中的连接配置

### 问题 3：密码错误

**错误信息**：`Authentication failed`

**解决方案**：
1. 确认密码为：`qianxi147A`
2. 如果忘记密码，可以重置：
   ```powershell
   # 停止 Neo4j
   .\neo4j.bat stop
   
   # 删除 auth 文件（会重置所有用户）
   del C:\neo4j\data\dbms\auth
   
   # 重新启动 Neo4j
   .\neo4j.bat start
   ```

### 问题 4：Java 版本问题

**错误信息**：`UnsupportedClassVersionError`

**解决方案**：
1. 检查 Java 版本：
   ```powershell
   java -version
   ```
2. 确保使用 Java 11 或更高版本
3. 下载并安装 Java：https://www.oracle.com/java/technologies/downloads/

---

## 📝 项目配置说明

### 当前项目配置

根据项目代码，当前配置如下：

```python
URI = "bolt://127.0.0.1:8687"  # 注意：端口是 8687，不是默认的 7687
USER = "neo4j"
PASSWORD = "qianxi147A"
DATABASE = "neo4j"  # 默认数据库
```

### 修改配置

如果需要修改配置，请同时更新以下文件：
- `kg/base.py`
- `kg/neo4j_tools.py`
- `kg/kg_agent.py`
- `kg/langchain_agent.py`

---

## 🚀 快速启动检查清单

- [ ] ✅ Neo4j 已安装并启动
- [ ] ✅ 端口配置为 8687
- [ ] ✅ 密码设置为 `qianxi147A`
- [ ] ✅ Python 连接测试通过（运行 `base.py`）
- [ ] ✅ 知识图谱已初始化（执行 `创建.cypher`）
- [ ] ✅ 数据验证通过（执行查询脚本）

---

## 📚 相关文档

- [`创建.cypher`](./创建.cypher) - 知识图谱创建脚本
- [`查询.cypher`](./查询.cypher) - Graph 查询脚本
- [`常用命令.cypher`](./常用命令.cypher) - 常用命令脚本
- [`关系.md`](./关系.md) - 知识图谱结构说明
- [`使用指南.md`](./使用指南.md) - Agent 使用指南

---

## 🔗 参考资源

- **Neo4j 官方文档**：https://neo4j.com/docs/
- **Neo4j Cypher 手册**：https://neo4j.com/docs/cypher-manual/
- **Neo4j Python 驱动**：https://neo4j.com/docs/python-manual/

---

*最后更新：2025年*

