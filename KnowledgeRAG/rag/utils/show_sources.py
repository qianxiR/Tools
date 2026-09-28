"""
检索来源显示工具 - 格式化显示 AI 从知识库检索到的文档信息
"""
import os

def format_source_path(source_path):
    """
    入参:
    - source_path (str): 文档的完整路径
    方法:
    - 格式化文档路径，提取相对路径和文件名
    - 将绝对路径转换为相对路径显示
    出参:
    - formatted_path (str): 格式化后的路径字符串
    """
    if not source_path or source_path == "未知来源":
        return "未知来源"
    
    # 提取文件名
    filename = os.path.basename(source_path)
    
    # 尝试提取相对路径（相对于项目根目录）
    # 查找常见的知识库目录标识
    if "solution_cases" in source_path:
        relative_path = source_path.split("solution_cases")[-1].lstrip(os.sep)
        return f"solution_cases/{relative_path}"
    elif "knowledge" in source_path:
        relative_path = source_path.split("knowledge")[-1].lstrip(os.sep)
        return f"knowledge/{relative_path}"
    elif "utils" in source_path:
        relative_path = source_path.split("utils")[-1].lstrip(os.sep)
        return f"utils/{relative_path}"
    else:
        # 如果无法识别，返回文件名
        return filename

def show_retrieved_sources(docs, query=None):
    """
    入参:
    - docs (list[Document]): 检索到的文档列表
    - query (str, optional): 查询文本，用于显示上下文
    方法:
    - 格式化并显示检索到的文档来源信息
    - 显示文档路径和具体文本片段
    - 支持显示相关性分数（如果有）
    出参:
    - 无（直接打印输出）
    """
    if not docs:
        print("\n📚 检索来源：未检索到相关文档")
        return
    
    print("\n" + "=" * 60)
    print("📚 检索来源信息")
    print("=" * 60)
    
    if query:
        print(f"查询: {query}\n")
    
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        content = doc.page_content
        
        # 格式化路径
        formatted_source = format_source_path(source)
        
        # 显示文档信息
        print(f"\n[文档 {i}]")
        print(f"📄 来源: {formatted_source}")
        
        # 如果有相关性分数，显示分数
        if hasattr(doc, 'metadata') and 'relevance_score' in doc.metadata:
            score = doc.metadata['relevance_score']
            print(f"⭐ 相关性分数: {score:.3f}")
        
        # 显示文本片段（限制长度，避免过长）
        max_length = 300
        if len(content) > max_length:
            display_content = content[:max_length] + "..."
        else:
            display_content = content
        
        print(f"📝 内容片段:")
        print(f"   {display_content}")
        
        # 如果内容被截断，显示总长度
        if len(content) > max_length:
            print(f"   ... (总长度: {len(content)} 字符)")
    
    print("\n" + "=" * 60)

def show_retrieved_sources_compact(docs):
    """
    入参:
    - docs (list[Document]): 检索到的文档列表
    方法:
    - 紧凑格式显示检索来源（仅显示文档路径）
    - 适合在对话流中快速显示
    出参:
    - 无（直接打印输出）
    """
    if not docs:
        print("\n📚 检索来源：未检索到相关文档")
        return
    
    print("\n📚 检索来源:")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        formatted_source = format_source_path(source)
        print(f"  [{i}] {formatted_source}")

