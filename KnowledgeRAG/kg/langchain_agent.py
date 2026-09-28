"""
基于 LangChain 和 Qwen-Plus 的知识图谱对话 Agent
"""
from langchain_community.chat_models import ChatTongyi
from neo4j_tools import Neo4jTools
import os
import json


class KnowledgeGraphTools:
    """知识图谱工具包装类"""
    
    def __init__(self, neo4j_tools):
        self.neo4j_tools = neo4j_tools
    
    def get_schema(self, query: str = "") -> str:
        """获取知识图谱结构"""
        result = self.neo4j_tools.get_schema_info()
        return f"""知识图谱结构:
- 节点总数: {result['node_count']}
- 关系总数: {result['relationship_count']}
- 节点类型: {', '.join(result['node_labels'])}
- 关系类型: {', '.join(result['relationship_types'][:10])}"""
    
    def get_entities(self, query: str = "") -> str:
        """获取所有实体"""
        entities = self.neo4j_tools.get_nodes_by_label("实体", limit=50)
        result = ["实体列表:"]
        for e in entities:
            name = e['properties'].get('名称', '未命名')
            result.append(f"- {name}")
        return '\n'.join(result)
    
    def get_indices(self, query: str = "") -> str:
        """获取所有监测指数"""
        indices = self.neo4j_tools.get_nodes_by_label("指数", limit=50)
        result = ["监测指数列表:"]
        for idx in indices:
            props = idx['properties']
            name = props.get('名称', '未命名')
            meaning = props.get('含义', '')
            result.append(f"- {name}: {meaning}")
        return '\n'.join(result)
    
    def get_states(self, query: str = "") -> str:
        """获取所有状态阶段"""
        states = self.neo4j_tools.get_nodes_by_label("状态", limit=50)
        result = ["滑坡恢复阶段:"]
        sorted_states = sorted(states, key=lambda x: x['properties'].get('顺序', 999))
        for s in sorted_states:
            props = s['properties']
            name = props.get('名称', '未命名')
            order = props.get('顺序', '?')
            meaning = props.get('语义', '')
            result.append(f"- 阶段{order}: {name} ({meaning})")
        return '\n'.join(result)
    
    def query_entity_relations(self, entity_name: str) -> str:
        """查询实体的关系"""
        node = self.neo4j_tools.get_node_by_property("实体", "名称", entity_name)
        if not node:
            return f"未找到实体: {entity_name}"
        
        relations = self.neo4j_tools.get_node_relationships("实体", "名称", entity_name, "both")
        
        result = [f"实体'{entity_name}'的关系:"]
        for r in relations[:20]:
            rel_type = r['type']
            node_b_name = r['node_b']['properties'].get('名称', '未知')
            result.append(f"- {entity_name} --[{rel_type}]--> {node_b_name}")
        
        return '\n'.join(result)
    
    def execute_cypher(self, query: str) -> str:
        """执行 Cypher 查询"""
        try:
            result = self.neo4j_tools.execute_cypher(query, {})
            if not result:
                return "查询未返回结果"
            
            output = ["查询结果:"]
            for i, record in enumerate(result[:10], 1):
                output.append(f"{i}. {record}")
            
            if len(result) > 10:
                output.append(f"... 还有 {len(result)-10} 条结果")
            
            return '\n'.join(output)
        except Exception as e:
            return f"查询出错: {str(e)}"


class SimpleKGAgent:
    """简化版知识图谱 Agent"""
    
    def __init__(self, llm, kg_tools):
        self.llm = llm
        self.kg_tools = kg_tools
        
        # 定义系统提示
        self.system_prompt = """你是一个知识图谱助手，专门帮助用户查询和理解震后滑坡生态恢复知识图谱。

知识图谱包含:
- 实体: 土壤、植被、滑坡等
- 监测指数: NDVI、NDWI、NBR等
- 恢复阶段: 滑坡前、滑坡后、恢复中、已恢复
- 各种关系: 监测指标、包含实体、对应阶段等

你可以使用以下工具来获取信息:
1. get_schema() - 获取知识图谱的整体结构
2. get_entities() - 获取所有实体列表
3. get_indices() - 获取所有监测指数
4. get_states() - 获取恢复阶段
5. query_entity_relations(实体名称) - 查询某个实体的关系

请根据用户的问题，选择合适的工具获取信息，然后用清晰友好的自然语言回答用户。
如果用户问"你是谁"，介绍你是知识图谱分析助手。"""
    
    def stream(self, query: str):
        """流式处理用户查询"""
        # 简单的意图识别
        query_lower = query.lower()
        
        # 如果是打招呼或询问身份
        if any(word in query_lower for word in ['你是', '你好', 'hello', 'hi']):
            context = f"用户问: {query}\n\n你是知识图谱分析助手，请友好地介绍自己。"
        
        # 如果询问结构或概览
        elif any(word in query for word in ['结构', '有什么', '概况', '多少']):
            schema_info = self.kg_tools.get_schema()
            context = f"用户问: {query}\n\n知识图谱结构信息:\n{schema_info}\n\n请用自然语言解释这些信息。"
        
        # 如果询问实体
        elif '实体' in query:
            entities_info = self.kg_tools.get_entities()
            context = f"用户问: {query}\n\n{entities_info}\n\n请用自然语言介绍这些实体。"
        
        # 如果询问指数
        elif any(word in query for word in ['指数', '指标', '监测']):
            indices_info = self.kg_tools.get_indices()
            context = f"用户问: {query}\n\n{indices_info}\n\n请用自然语言介绍这些监测指数。"
        
        # 如果询问状态或阶段
        elif any(word in query for word in ['状态', '阶段', '恢复', '过程']):
            states_info = self.kg_tools.get_states()
            context = f"用户问: {query}\n\n{states_info}\n\n请用自然语言介绍恢复过程。"
        
        # 如果询问特定实体
        elif any(entity in query for entity in ['植被', '土壤', '滑坡', '水体']):
            for entity in ['植被', '土壤', '滑坡', '水体']:
                if entity in query:
                    relations_info = self.kg_tools.query_entity_relations(entity)
                    context = f"用户问: {query}\n\n{relations_info}\n\n请用自然语言解释{entity}的相关信息。"
                    break
        else:
            # 默认获取结构信息
            schema_info = self.kg_tools.get_schema()
            context = f"用户问: {query}\n\n知识图谱结构:\n{schema_info}\n\n请根据用户问题回答。"
        
        # 流式调用 LLM
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": context}
        ]
        
        # 返回流式生成器
        for chunk in self.llm.stream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content


def create_kg_agent(api_key: str):
    """创建知识图谱 Agent"""
    
    # 连接 Neo4j
    URI = "bolt://127.0.0.1:8687"
    USER = "neo4j"
    PASSWORD = "qianxi147A"
    
    neo4j_tools = Neo4jTools(URI, USER, PASSWORD)
    kg_tools = KnowledgeGraphTools(neo4j_tools)
    
    # 创建 LLM
    llm = ChatTongyi(
        model="qwen-plus",
        dashscope_api_key=api_key,
        temperature=0.5
    )
    
    # 创建简化 Agent
    agent = SimpleKGAgent(llm, kg_tools)
    
    return agent, neo4j_tools


def chat_loop():
    """对话循环 - 支持流式输出"""
    
    # 获取 API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        api_key = input("请输入您的 DashScope API Key: ").strip()
    
    print("\n" + "="*100)
    print("🤖 知识图谱智能对话 Agent (基于 Qwen-Plus) - 流式对话模式")
    print("="*100)
    print("输入 'exit' 或 'quit' 退出")
    print("="*100 + "\n")
    
    try:
        agent, neo4j_tools = create_kg_agent(api_key)
        
        while True:
            try:
                user_input = input("\n👤 您: ").strip()
                
                if user_input.lower() in ['exit', 'quit', '退出']:
                    print("\n👋 再见!")
                    break
                
                if not user_input:
                    continue
                
                print("\n🤖 Agent: ", end="", flush=True)
                
                # 流式调用 Agent
                for chunk in agent.stream(user_input):
                    print(chunk, end="", flush=True)
                
                print()  # 换行
                
            except KeyboardInterrupt:
                print("\n\n👋 再见!")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")
                import traceback
                traceback.print_exc()
        
        # 关闭连接
        neo4j_tools.close()
        
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    chat_loop()
