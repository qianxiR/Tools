"""
主程序 - 知识库驱动的智能 Agent
"""
from utils.embeding import create_vector_store, search_knowledge_base
from utils.show_sources import show_retrieved_sources
from rag_agent import create_agent_with_memory, create_simple_rag_agent

# 全局变量配置
FORCE_RECREATE_VECTOR_DB = False  # 是否强制重新创建向量数据库（True: 每次启动都重建，False: 如果存在则直接加载）
USE_RERANK = True  # 是否启用重排序功能

def main():
    """
    入参:
    - 无
    方法:
    - 主函数，初始化知识库和 Agent
    - 提供交互式聊天界面
    - 支持知识库检索和问答
    出参:
    - 无
    """
    print("=" * 60)
    print("知识库驱动的智能 Agent")
    print("=" * 60)
    
    # 初始化向量数据库
    print("\n正在初始化知识库...")
    try:
        vectorstore = create_vector_store(force_recreate=FORCE_RECREATE_VECTOR_DB)
        print("知识库初始化完成！\n")
    except Exception as e:
        print(f"知识库初始化失败: {e}")
        return
    
    # 创建 Agent（带记忆版本，启用重排序）
    print("正在创建 Agent...")
    from utils.embeding import USE_RERANK
    agent = create_agent_with_memory(vectorstore, use_rerank=USE_RERANK)
    print(f"Agent 创建完成！{'（已启用重排序）' if USE_RERANK else ''}\n")
    
    # 显示使用说明
    print("=" * 60)
    print("使用说明:")
    print("- 输入问题，Agent 会从知识库中检索相关信息并回答")
    print("- 输入 'clear' 清空对话历史")
    print("- 输入 'search <关键词>' 直接搜索知识库")
    print("- 输入 'quit' 或 'exit' 退出")
    print("=" * 60)
    print()
    
    # 交互式对话循环
    while True:
        try:
            user_input = input("你: ").strip()
            
            if not user_input:
                continue
            
            # 退出命令
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break
            
            # 清空历史命令
            if user_input.lower() == 'clear':
                agent.clear_history()
                print("对话历史已清空\n")
                continue
            
            # 直接搜索知识库
            if user_input.lower().startswith('search '):
                query = user_input[7:].strip()
                print(f"\n搜索知识库: {query}")
                from utils.embeding import USE_RERANK
                results = search_knowledge_base(vectorstore, query, k=3, use_rerank=USE_RERANK)
                print(f"\n找到 {len(results)} 个相关文档:\n")
                for i, doc in enumerate(results, 1):
                    source = doc.metadata.get("source", "未知来源")
                    print(f"[{i}] 来源: {source}")
                    print(f"内容: {doc.page_content[:200]}...")
                    print()
                continue
            
            # 正常对话（流式输出，自动判断是否需要检索知识库）
            print("助手: ", end="", flush=True)
            # 使用流式输出，use_rag="auto" 表示自动判断是否需要检索
            for chunk in agent.chat_stream(user_input, use_rag="auto"):
                print(chunk, end="", flush=True)
            print()
            
            # 显示检索来源信息（对话结束后）
            retrieved_docs = agent.last_retrieved_docs
            if retrieved_docs:
                show_retrieved_sources(retrieved_docs, query=user_input)
            else:
                print("\n💡 提示：本次对话未检索知识库（使用通用知识回答）")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\n程序已中断")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            print("请重试或输入 'quit' 退出\n")

def demo():
    """
    入参:
    - 无
    方法:
    - 演示函数，展示知识库检索和问答功能
    - 运行预设的示例问题
    出参:
    - 无
    """
    print("=" * 60)
    print("知识库 Agent 演示")
    print("=" * 60)
    
    # 初始化知识库
    print("\n正在初始化知识库...")
    vectorstore = create_vector_store(force_recreate=FORCE_RECREATE_VECTOR_DB)
    print("知识库初始化完成！\n")
    
    # 创建简单 RAG Agent
    rag_chain = create_simple_rag_agent(vectorstore)
    
    # 示例问题
    demo_questions = [
        "如何进行水深反演分析？",
        "如何计算 NDVI 植被指数？",
        "如何使用 @ 符号标记图层？",
        "如何查询用户记忆？"
    ]
    
    print("=" * 60)
    print("开始演示问答...")
    print("=" * 60)
    print()
    
    for question in demo_questions:
        print(f"问题: {question}")
        print("回答: ", end="", flush=True)
        
        try:
            response = rag_chain.invoke(question)
            print(response)
        except Exception as e:
            print(f"错误: {e}")
        
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        main()

