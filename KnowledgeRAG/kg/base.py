from neo4j import GraphDatabase

# ======== 配置信息 ========
# 注意:Neo4j 默认 Bolt 端口是 7687,如果您修改过请相应调整
URI = "bolt://127.0.0.1:8687"  # 改为默认端口
USER = "neo4j"
PASSWORD = "qianxi147A"  # 请确认密码是否正确

# ======== 建立连接 ========
try:
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    
    # 验证连接
    driver.verify_connectivity()
    print("✅ 成功连接到 Neo4j!")
    
    with driver.session(database="neo4j") as session:
        print("\n📘 查询所有节点")
        result_nodes = session.run("MATCH (n) RETURN ID(n) AS id, labels(n) AS labels, n LIMIT 20")
        for record in result_nodes:
            print(f"节点ID={record['id']}, 标签={record['labels']}, 属性={dict(record['n'])}")

        print("\n🔗 查询所有关系")
        result_rels = session.run("""
            MATCH (a)-[r]->(b)
            RETURN ID(r) AS id, type(r) AS type, a.name AS source, b.name AS target
            LIMIT 20
        """)
        for record in result_rels:
            print(f"关系ID={record['id']}, 类型={record['type']}, {record['source']} → {record['target']}")
    
    driver.close()
    
except Exception as e:
    print(f"❌ 连接失败: {e}")
    print("\n请检查:")
    print("1. Neo4j 服务是否正在运行")
    print("2. URI 端口是否正确 (默认 7687)")
    print("3. 用户名和密码是否正确")
    print("4. 如果是首次使用,可能需要修改默认密码")
