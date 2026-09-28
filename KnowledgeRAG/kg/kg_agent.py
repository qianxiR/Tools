"""
知识图谱 Agent
使用 Neo4j 工具来回答用户关于知识图谱的查询
"""
from typing import List, Dict, Any, Optional, Callable
import json
from neo4j_tools import Neo4jTools, TOOL_DESCRIPTIONS


class KnowledgeGraphAgent:
    """知识图谱智能代理,能够理解用户意图并调用合适的工具"""
    
    def __init__(self, neo4j_tools: Neo4jTools):
        """
        初始化 Agent
        
        Args:
            neo4j_tools: Neo4j 工具实例
        """
        self.tools = neo4j_tools
        self.available_tools = self._register_tools()
        self.conversation_history = []
        
    def _register_tools(self) -> Dict[str, Callable]:
        """注册所有可用的工具"""
        return {
            "get_all_nodes": self.tools.get_all_nodes,
            "get_nodes_by_label": self.tools.get_nodes_by_label,
            "get_all_relationships": self.tools.get_all_relationships,
            "get_node_by_property": self.tools.get_node_by_property,
            "get_node_relationships": self.tools.get_node_relationships,
            "find_path": self.tools.find_path,
            "get_schema_info": self.tools.get_schema_info,
            "execute_cypher": self.tools.execute_cypher,
        }
    
    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """获取所有工具的描述信息"""
        return list(TOOL_DESCRIPTIONS.values())
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """
        执行指定的工具
        
        Args:
            tool_name: 工具名称
            parameters: 工具参数
            
        Returns:
            工具执行结果
        """
        if tool_name not in self.available_tools:
            raise ValueError(f"工具 '{tool_name}' 不存在。可用工具: {list(self.available_tools.keys())}")
        
        tool_func = self.available_tools[tool_name]
        
        try:
            result = tool_func(**parameters)
            return {
                "success": True,
                "tool": tool_name,
                "parameters": parameters,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "tool": tool_name,
                "parameters": parameters,
                "error": str(e)
            }
    
    def query(self, user_query: str, tool_calls: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        处理用户查询
        
        Args:
            user_query: 用户的自然语言查询
            tool_calls: 要调用的工具列表,格式为 [{"tool": "工具名", "parameters": {...}}]
                       如果为 None,则需要先通过 LLM 解析用户意图
            
        Returns:
            查询结果
        """
        self.conversation_history.append({
            "role": "user",
            "content": user_query
        })
        
        if tool_calls is None:
            # 如果没有提供工具调用,返回提示信息
            return {
                "status": "需要工具调用",
                "message": "请使用 LLM 解析用户意图,并调用相应的工具",
                "user_query": user_query,
                "available_tools": self.get_tool_descriptions()
            }
        
        # 执行工具调用
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool")
            parameters = tool_call.get("parameters", {})
            
            result = self.execute_tool(tool_name, parameters)
            results.append(result)
        
        response = {
            "user_query": user_query,
            "tool_calls": tool_calls,
            "results": results
        }
        
        self.conversation_history.append({
            "role": "assistant",
            "content": json.dumps(response, ensure_ascii=False)
        })
        
        return response
    
    def format_response(self, query_result: Dict[str, Any]) -> str:
        """
        格式化查询结果为友好的文本输出
        
        Args:
            query_result: query 方法返回的结果
            
        Returns:
            格式化的文本
        """
        output = []
        output.append(f"📝 用户查询: {query_result['user_query']}\n")
        
        for i, result in enumerate(query_result.get('results', []), 1):
            output.append(f"--- 工具调用 {i} ---")
            output.append(f"🔧 工具: {result['tool']}")
            output.append(f"📋 参数: {json.dumps(result['parameters'], ensure_ascii=False)}")
            
            if result['success']:
                output.append(f"✅ 执行成功")
                output.append(f"📊 结果:")
                
                data = result['result']
                if isinstance(data, list):
                    output.append(f"   找到 {len(data)} 条记录")
                    for idx, item in enumerate(data[:5], 1):  # 只显示前5条
                        output.append(f"   {idx}. {json.dumps(item, ensure_ascii=False, indent=6)}")
                    if len(data) > 5:
                        output.append(f"   ... 还有 {len(data) - 5} 条记录")
                elif isinstance(data, dict):
                    output.append(json.dumps(data, ensure_ascii=False, indent=3))
                else:
                    output.append(f"   {data}")
            else:
                output.append(f"❌ 执行失败: {result['error']}")
            
            output.append("")
        
        return "\n".join(output)
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.conversation_history
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []


# ==================== 预定义查询模板 ====================

class QueryTemplates:
    """常见查询的模板"""
    
    @staticmethod
    def get_overview():
        """获取知识图谱概览"""
        return [
            {"tool": "get_schema_info", "parameters": {}}
        ]
    
    @staticmethod
    def get_all_entities(limit: int = 100):
        """获取所有实体"""
        return [
            {"tool": "get_nodes_by_label", "parameters": {"label": "实体", "limit": limit}}
        ]
    
    @staticmethod
    def get_all_states(limit: int = 100):
        """获取所有状态"""
        return [
            {"tool": "get_nodes_by_label", "parameters": {"label": "状态", "limit": limit}}
        ]
    
    @staticmethod
    def get_all_indices(limit: int = 100):
        """获取所有指数"""
        return [
            {"tool": "get_nodes_by_label", "parameters": {"label": "指数", "limit": limit}}
        ]
    
    @staticmethod
    def get_entity_info(entity_name: str):
        """获取特定实体的详细信息"""
        return [
            {"tool": "get_node_by_property", 
             "parameters": {"label": "实体", "property_name": "名称", "property_value": entity_name}},
            {"tool": "get_node_relationships",
             "parameters": {"node_label": "实体", "node_property": "名称", 
                           "node_value": entity_name, "direction": "both"}}
        ]
    
    @staticmethod
    def find_relationship_path(start_entity: str, end_entity: str, max_depth: int = 5):
        """查找两个实体之间的关系路径"""
        return [
            {"tool": "find_path",
             "parameters": {
                 "start_label": "实体",
                 "start_property": "名称",
                 "start_value": start_entity,
                 "end_label": "实体",
                 "end_property": "名称",
                 "end_value": end_entity,
                 "max_depth": max_depth
             }}
        ]
    
    @staticmethod
    def get_state_transitions():
        """获取状态转换关系"""
        return [
            {"tool": "execute_cypher",
             "parameters": {
                 "query": """
                     MATCH (s1:状态)-[r]->(s2:状态)
                     RETURN s1.名称 AS from_state, type(r) AS relation, s2.名称 AS to_state,
                            s1.顺序 AS from_order, s2.顺序 AS to_order
                     ORDER BY s1.顺序
                 """
             }}
        ]


# ==================== 使用示例 ====================

def example_usage():
    """Agent 使用示例"""
    
    # 配置信息
    URI = "bolt://127.0.0.1:8687"
    USER = "neo4j"
    PASSWORD = "qianxi147A"
    
    # 创建工具和 Agent
    with Neo4jTools(URI, USER, PASSWORD) as neo4j_tools:
        agent = KnowledgeGraphAgent(neo4j_tools)
        
        print("=" * 80)
        print("🤖 知识图谱 Agent 示例")
        print("=" * 80)
        
        # 示例 1: 获取知识图谱概览
        print("\n【示例 1】获取知识图谱概览")
        print("-" * 80)
        result1 = agent.query(
            "请告诉我知识图谱的整体结构",
            tool_calls=QueryTemplates.get_overview()
        )
        print(agent.format_response(result1))
        
        # 示例 2: 查询所有实体
        print("\n【示例 2】查询所有实体")
        print("-" * 80)
        result2 = agent.query(
            "列出所有的实体",
            tool_calls=QueryTemplates.get_all_entities()
        )
        print(agent.format_response(result2))
        
        # 示例 3: 查询特定实体的信息
        print("\n【示例 3】查询'植被'实体的详细信息")
        print("-" * 80)
        result3 = agent.query(
            "告诉我关于'植被'的详细信息",
            tool_calls=QueryTemplates.get_entity_info("植被")
        )
        print(agent.format_response(result3))
        
        # 示例 4: 查询所有指数
        print("\n【示例 4】查询所有指数")
        print("-" * 80)
        result4 = agent.query(
            "有哪些指数可以用于分析?",
            tool_calls=QueryTemplates.get_all_indices()
        )
        print(agent.format_response(result4))
        
        # 示例 5: 查询状态转换
        print("\n【示例 5】查询状态转换关系")
        print("-" * 80)
        result5 = agent.query(
            "滑坡的状态是如何演变的?",
            tool_calls=QueryTemplates.get_state_transitions()
        )
        print(agent.format_response(result5))
        
        # 示例 6: 自定义 Cypher 查询
        print("\n【示例 6】查询所有时相信息")
        print("-" * 80)
        result6 = agent.query(
            "列出所有的时相节点",
            tool_calls=[{
                "tool": "get_nodes_by_label",
                "parameters": {"label": "时相", "limit": 50}
            }]
        )
        print(agent.format_response(result6))


if __name__ == "__main__":
    example_usage()
