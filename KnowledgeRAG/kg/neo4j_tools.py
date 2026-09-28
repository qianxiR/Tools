"""
Neo4j 知识图谱工具封装
提供查询节点、关系、路径等功能
"""
from neo4j import GraphDatabase
from neo4j.time import Date, DateTime, Time, Duration
from typing import List, Dict, Any, Optional
import json
from datetime import date, datetime


def convert_neo4j_types(obj):
    """
    递归转换 Neo4j 特殊类型为 Python 原生类型
    
    Args:
        obj: 任意对象
        
    Returns:
        转换后的对象
    """
    if isinstance(obj, (Date, DateTime)):
        return obj.iso_format()
    elif isinstance(obj, Time):
        return str(obj)
    elif isinstance(obj, Duration):
        return {
            "months": obj.months,
            "days": obj.days,
            "seconds": obj.seconds,
            "nanoseconds": obj.nanoseconds
        }
    elif isinstance(obj, (date, datetime)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: convert_neo4j_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_neo4j_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_neo4j_types(item) for item in obj)
    else:
        return obj


class Neo4jTools:
    """Neo4j 工具类,封装常用的图数据库操作"""
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """
        初始化 Neo4j 连接
        
        Args:
            uri: Neo4j 连接地址 (例如: bolt://127.0.0.1:8687)
            user: 用户名
            password: 密码
            database: 数据库名称
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self._connect()
    
    def _connect(self):
        """建立连接"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            print("✅ Neo4j 连接成功!")
        except Exception as e:
            print(f"❌ Neo4j 连接失败: {e}")
            raise
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            print("🔌 Neo4j 连接已关闭")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # ==================== 工具方法 ====================
    
    def get_all_nodes(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取所有节点
        
        Args:
            limit: 返回的最大节点数
            
        Returns:
            节点列表,每个节点包含 id, labels, properties
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(
                "MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties LIMIT $limit",
                limit=limit
            )
            nodes = []
            for record in result:
                nodes.append({
                    "id": record["id"],
                    "labels": record["labels"],
                    "properties": convert_neo4j_types(dict(record["properties"]))
                })
            return nodes
    
    def get_nodes_by_label(self, label: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        根据标签获取节点
        
        Args:
            label: 节点标签 (例如: "实体", "状态", "指数")
            limit: 返回的最大节点数
            
        Returns:
            节点列表
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"MATCH (n:{label}) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties LIMIT $limit",
                limit=limit
            )
            nodes = []
            for record in result:
                nodes.append({
                    "id": record["id"],
                    "labels": record["labels"],
                    "properties": convert_neo4j_types(dict(record["properties"]))
                })
            return nodes
    
    def get_all_relationships(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取所有关系
        
        Args:
            limit: 返回的最大关系数
            
        Returns:
            关系列表,每个关系包含 id, type, source, target, properties
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (a)-[r]->(b)
                RETURN elementId(r) AS id, type(r) AS type, 
                       elementId(a) AS source_id, labels(a) AS source_labels, properties(a) AS source_props,
                       elementId(b) AS target_id, labels(b) AS target_labels, properties(b) AS target_props,
                       properties(r) AS rel_properties
                LIMIT $limit
            """, limit=limit)
            
            relationships = []
            for record in result:
                relationships.append({
                    "id": record["id"],
                    "type": record["type"],
                    "source": {
                        "id": record["source_id"],
                        "labels": record["source_labels"],
                        "properties": convert_neo4j_types(dict(record["source_props"]))
                    },
                    "target": {
                        "id": record["target_id"],
                        "labels": record["target_labels"],
                        "properties": convert_neo4j_types(dict(record["target_props"]))
                    },
                    "properties": convert_neo4j_types(dict(record["rel_properties"]))
                })
            return relationships
    
    def get_node_by_property(self, label: str, property_name: str, property_value: Any) -> Optional[Dict[str, Any]]:
        """
        根据属性查找节点
        
        Args:
            label: 节点标签
            property_name: 属性名
            property_value: 属性值
            
        Returns:
            节点信息或 None
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"MATCH (n:{label} {{{property_name}: $value}}) "
                f"RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties LIMIT 1",
                value=property_value
            )
            record = result.single()
            if record:
                return {
                    "id": record["id"],
                    "labels": record["labels"],
                    "properties": convert_neo4j_types(dict(record["properties"]))
                }
            return None
    
    def get_node_relationships(self, node_label: str, node_property: str, node_value: Any, 
                                direction: str = "both") -> List[Dict[str, Any]]:
        """
        获取某个节点的所有关系
        
        Args:
            node_label: 节点标签
            node_property: 节点属性名
            node_value: 节点属性值
            direction: 关系方向 ("in", "out", "both")
            
        Returns:
            关系列表
        """
        direction_pattern = {
            "in": "<-[r]-(b)",
            "out": "-[r]->(b)",
            "both": "-[r]-(b)"
        }
        
        pattern = direction_pattern.get(direction, "-[r]-(b)")
        
        with self.driver.session(database=self.database) as session:
            result = session.run(f"""
                MATCH (a:{node_label} {{{node_property}: $value}}){pattern}
                RETURN elementId(r) AS id, type(r) AS type,
                       elementId(a) AS node_a_id, labels(a) AS node_a_labels, properties(a) AS node_a_props,
                       elementId(b) AS node_b_id, labels(b) AS node_b_labels, properties(b) AS node_b_props,
                       properties(r) AS rel_properties
            """, value=node_value)
            
            relationships = []
            for record in result:
                relationships.append({
                    "id": record["id"],
                    "type": record["type"],
                    "node_a": {
                        "id": record["node_a_id"],
                        "labels": record["node_a_labels"],
                        "properties": convert_neo4j_types(dict(record["node_a_props"]))
                    },
                    "node_b": {
                        "id": record["node_b_id"],
                        "labels": record["node_b_labels"],
                        "properties": convert_neo4j_types(dict(record["node_b_props"]))
                    },
                    "properties": convert_neo4j_types(dict(record["rel_properties"]))
                })
            return relationships
    
    def find_path(self, start_label: str, start_property: str, start_value: Any,
                  end_label: str, end_property: str, end_value: Any,
                  max_depth: int = 5) -> List[Dict[str, Any]]:
        """
        查找两个节点之间的路径
        
        Args:
            start_label: 起始节点标签
            start_property: 起始节点属性名
            start_value: 起始节点属性值
            end_label: 终止节点标签
            end_property: 终止节点属性名
            end_value: 终止节点属性值
            max_depth: 最大搜索深度
            
        Returns:
            路径列表
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(f"""
                MATCH path = (start:{start_label} {{{start_property}: $start_value}})
                             -[*1..{max_depth}]-
                             (end:{end_label} {{{end_property}: $end_value}})
                RETURN path
                LIMIT 10
            """, start_value=start_value, end_value=end_value)
            
            paths = []
            for record in result:
                path = record["path"]
                path_info = {
                    "length": len(path.relationships),
                    "nodes": [],
                    "relationships": []
                }
                
                # 提取节点
                for node in path.nodes:
                    path_info["nodes"].append({
                        "labels": list(node.labels),
                        "properties": convert_neo4j_types(dict(node))
                    })
                
                # 提取关系
                for rel in path.relationships:
                    path_info["relationships"].append({
                        "type": rel.type,
                        "properties": convert_neo4j_types(dict(rel))
                    })
                
                paths.append(path_info)
            
            return paths
    
    def execute_cypher(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        执行自定义 Cypher 查询
        
        Args:
            query: Cypher 查询语句
            parameters: 查询参数
            
        Returns:
            查询结果列表
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            records = []
            for record in result:
                records.append(convert_neo4j_types(dict(record)))
            return records
    
    def get_schema_info(self) -> Dict[str, Any]:
        """
        获取图数据库的模式信息
        
        Returns:
            包含节点标签、关系类型等信息的字典
        """
        with self.driver.session(database=self.database) as session:
            # 获取所有节点标签
            labels_result = session.run("CALL db.labels()")
            labels = [record[0] for record in labels_result]
            
            # 获取所有关系类型
            types_result = session.run("CALL db.relationshipTypes()")
            relationship_types = [record[0] for record in types_result]
            
            # 获取节点数量
            node_count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
            
            # 获取关系数量
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
            
            return {
                "node_labels": labels,
                "relationship_types": relationship_types,
                "node_count": node_count,
                "relationship_count": rel_count
            }


# ==================== 工具函数描述 (用于 Agent) ====================

TOOL_DESCRIPTIONS = {
    "get_all_nodes": {
        "name": "get_all_nodes",
        "description": "获取知识图谱中的所有节点信息,返回节点的ID、标签和属性",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "返回的最大节点数量,默认100",
                    "default": 100
                }
            }
        }
    },
    "get_nodes_by_label": {
        "name": "get_nodes_by_label",
        "description": "根据标签查询特定类型的节点,例如查询所有'实体'、'状态'、'指数'等类型的节点",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "节点标签,如'实体'、'状态'、'指数'、'时相'、'空间区域'等",
                    "required": True
                },
                "limit": {
                    "type": "integer",
                    "description": "返回的最大节点数量,默认100",
                    "default": 100
                }
            },
            "required": ["label"]
        }
    },
    "get_all_relationships": {
        "name": "get_all_relationships",
        "description": "获取知识图谱中的所有关系,包含关系的类型、起始节点和目标节点信息",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "返回的最大关系数量,默认100",
                    "default": 100
                }
            }
        }
    },
    "get_node_by_property": {
        "name": "get_node_by_property",
        "description": "根据节点的属性查找特定节点,例如通过'名称'查找某个实体",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "节点标签",
                    "required": True
                },
                "property_name": {
                    "type": "string",
                    "description": "属性名称,如'名称'、'含义'等",
                    "required": True
                },
                "property_value": {
                    "type": "string",
                    "description": "属性值",
                    "required": True
                }
            },
            "required": ["label", "property_name", "property_value"]
        }
    },
    "get_node_relationships": {
        "name": "get_node_relationships",
        "description": "获取某个特定节点的所有关系,可以指定查询入边、出边或所有关系",
        "parameters": {
            "type": "object",
            "properties": {
                "node_label": {
                    "type": "string",
                    "description": "节点标签",
                    "required": True
                },
                "node_property": {
                    "type": "string",
                    "description": "节点属性名",
                    "required": True
                },
                "node_value": {
                    "type": "string",
                    "description": "节点属性值",
                    "required": True
                },
                "direction": {
                    "type": "string",
                    "description": "关系方向: 'in'(入边)、'out'(出边)、'both'(双向)",
                    "default": "both",
                    "enum": ["in", "out", "both"]
                }
            },
            "required": ["node_label", "node_property", "node_value"]
        }
    },
    "find_path": {
        "name": "find_path",
        "description": "查找两个节点之间的路径,可用于分析节点之间的关联关系",
        "parameters": {
            "type": "object",
            "properties": {
                "start_label": {
                    "type": "string",
                    "description": "起始节点标签",
                    "required": True
                },
                "start_property": {
                    "type": "string",
                    "description": "起始节点属性名",
                    "required": True
                },
                "start_value": {
                    "type": "string",
                    "description": "起始节点属性值",
                    "required": True
                },
                "end_label": {
                    "type": "string",
                    "description": "终止节点标签",
                    "required": True
                },
                "end_property": {
                    "type": "string",
                    "description": "终止节点属性名",
                    "required": True
                },
                "end_value": {
                    "type": "string",
                    "description": "终止节点属性值",
                    "required": True
                },
                "max_depth": {
                    "type": "integer",
                    "description": "最大搜索深度,默认5",
                    "default": 5
                }
            },
            "required": ["start_label", "start_property", "start_value", 
                        "end_label", "end_property", "end_value"]
        }
    },
    "get_schema_info": {
        "name": "get_schema_info",
        "description": "获取知识图谱的整体结构信息,包括所有节点标签、关系类型、节点和关系的数量统计",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    "execute_cypher": {
        "name": "execute_cypher",
        "description": "执行自定义的Cypher查询语句,适用于复杂的图查询需求",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Cypher查询语句",
                    "required": True
                },
                "parameters": {
                    "type": "object",
                    "description": "查询参数字典",
                    "default": {}
                }
            },
            "required": ["query"]
        }
    }
}
