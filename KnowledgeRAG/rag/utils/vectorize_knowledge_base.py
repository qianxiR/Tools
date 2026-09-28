"""
知识库向量化脚本

入参:
- 无
方法:
- 从三个知识库目录（solution_cases、knowledge、utils）加载文档
- 将文档分割成块
- 使用嵌入模型向量化
- 存储到 ChromaDB 向量数据库
出参:
- 无
"""
# 导入方式：从 rag/ 目录运行时使用绝对导入
# 如果从 utils/ 目录运行，需要先添加父目录到路径
import sys
import os
# 添加父目录到路径，以便导入 utils.embeding
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.embeding import create_vector_store, load_documents, KNOWLEDGE_BASE_DIRS

# 全局变量配置
FORCE_RECREATE = True  # 是否强制重新创建向量数据库

def main():
    """
    入参:
    - 无
    方法:
    - 主函数，执行知识库向量化流程
    - 显示详细的进度信息
    出参:
    - 无
    """
    print("=" * 60)
    print("知识库向量化工具")
    print("=" * 60)
    print()
    
    # 显示知识库目录
    print("📂 知识库目录配置：")
    for i, dir_name in enumerate(KNOWLEDGE_BASE_DIRS, 1):
        print(f"  {i}. {dir_name}/")
    print()
    
    # 检查目录是否存在
    import os
    print("🔍 检查知识库目录...")
    missing_dirs = []
    for dir_name in KNOWLEDGE_BASE_DIRS:
        if os.path.exists(dir_name):
            # 统计文件数量
            md_files = list(os.path.join(dir_name, f) for f in os.listdir(dir_name) 
                          if f.endswith('.md'))
            print(f"  ✅ {dir_name}/ - 找到 {len(md_files)} 个 Markdown 文件")
        else:
            print(f"  ❌ {dir_name}/ - 目录不存在")
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"\n⚠️  警告：以下目录不存在: {', '.join(missing_dirs)}")
        print("   这些目录将被跳过")
    
    print()
    
    # 加载文档
    print("📖 加载文档...")
    try:
        documents = load_documents()
        print(f"✅ 成功加载 {len(documents)} 个文档\n")
    except Exception as e:
        print(f"❌ 加载文档失败: {e}")
        return
    
    # 向量化
    print("🔄 开始向量化...")
    print(f"   配置：强制重建 = {FORCE_RECREATE}")
    print()
    
    try:
        vectorstore = create_vector_store(
            documents=documents,
            force_recreate=FORCE_RECREATE,
            interactive=True  # 交互模式，失败时询问用户
        )
        
        print()
        print("=" * 60)
        print("✅ 向量化完成！")
        print("=" * 60)
        print()
        print("📊 向量数据库信息：")
        print(f"   存储位置: ./chroma_db/")
        
        # 统计文档块数量
        try:
            collection_count = vectorstore._collection.count()
            print(f"   文档块数量: {collection_count}")
        except:
            print("   文档块数量: 无法获取")
        
        print()
        print("💡 提示：现在可以使用 main_agent.py 启动 RAG Agent")
        
    except Exception as e:
        print(f"\n❌ 向量化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

