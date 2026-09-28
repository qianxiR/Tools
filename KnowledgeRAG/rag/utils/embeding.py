"""
知识库管理模块 - 基于 LangChain 1.0 的 RAG 系统
"""
import os
import shutil
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# 全局变量配置
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")

# 获取当前文件所在目录的父目录（rag/ 目录）
_RAG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 知识库目录配置（支持多个目录）- 使用绝对路径，确保无论从哪里运行都能找到
KNOWLEDGE_BASE_DIRS = [
    os.path.join(_RAG_DIR, "space", "solution_cases"),
    os.path.join(_RAG_DIR, "space", "knowledge"),
    os.path.join(_RAG_DIR, "space", "tools")
]
VECTOR_DB_DIR = os.path.join(_RAG_DIR, "chroma_db")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 重排序配置
USE_RERANK = True  # 是否启用重排序
RERANK_TOP_K = 4  # 重排序后返回的文档数量
INITIAL_RETRIEVE_K = 10  # 初始检索的文档数量（用于重排序）

# 向量化配置
BATCH_SIZE = 50  # 批量处理文档块的大小（避免一次性处理太多导致连接中断）
RETRY_TIMES = 3  # 重试次数
RETRY_DELAY = 2  # 重试延迟（秒）
BATCH_DELAY = 0.5  # 批次之间的延迟（秒）

# 嵌入模型配置
# 可选值: "dashscope", "openai", "huggingface", "ollama"
EMBEDDING_PROVIDER = "dashscope"
# DashScope 模型: "text-embedding-v1", "text-embedding-v2", "text-embedding-v3"
# OpenAI 模型: "text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"
# HuggingFace 模型: "sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-large-zh-v1.5" 等
EMBEDDING_MODEL = "text-embedding-v3"
# OpenAI API Key (如果使用 OpenAI 嵌入模型)
OPENAI_API_KEY = ""
# HuggingFace 模型路径 (如果使用本地模型)
HUGGINGFACE_MODEL_PATH = ""
# Ollama 模型名称 (如果使用 Ollama)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "nomic-embed-text"

def load_documents():
    """
    入参:
    - 无
    方法:
    - 从多个知识库目录加载所有 Markdown 文档（solution_cases、knowledge、utils）
    - 使用 DirectoryLoader 自动识别 .md 文件
    - 合并所有目录的文档
    出参:
    - documents (list[Document]): 加载的文档列表
    """
    all_documents = []
    
    # 遍历所有知识库目录
    for base_dir in KNOWLEDGE_BASE_DIRS:
        if not os.path.exists(base_dir):
            print(f"警告：知识库目录不存在: {base_dir}，跳过")
            continue
        
        # 加载该目录下的所有 Markdown 文件
        loader = DirectoryLoader(
            base_dir,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True
        )
        
        documents = loader.load()
        all_documents.extend(documents)
        print(f"从 {base_dir} 加载了 {len(documents)} 个文档")
    
    if not all_documents:
        raise ValueError(f"没有找到任何文档，请检查知识库目录: {KNOWLEDGE_BASE_DIRS}")
    
    print(f"总共成功加载 {len(all_documents)} 个文档")
    
    return all_documents

def split_documents(documents):
    """
    入参:
    - documents (list[Document]): 原始文档列表
    方法:
    - 使用递归字符分割器将长文档切分为较小的块
    - 设置块大小和重叠大小，确保上下文连续性
    - 保留文档的元数据信息
    出参:
    - splits (list[Document]): 分割后的文档块列表
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", "。", " ", ""]
    )
    
    splits = text_splitter.split_documents(documents)
    print(f"文档分割完成，共 {len(splits)} 个文档块")
    
    return splits

def create_embeddings():
    """
    入参:
    - 无
    方法:
    - 根据配置的嵌入模型提供商创建对应的嵌入模型实例
    - 支持 DashScope、OpenAI、HuggingFace、Ollama 等多种嵌入服务
    - 自动设置必要的 API Key 和环境变量
    出参:
    - embeddings (Embeddings): 嵌入模型实例
    """
    provider = EMBEDDING_PROVIDER.lower()
    
    if provider == "dashscope":
        # DashScope 嵌入模型（阿里云通义千问）
        os.environ["DASHSCOPE_API_KEY"] = QWEN_API_KEY
        embeddings = DashScopeEmbeddings(model=EMBEDDING_MODEL)
        print(f"使用 DashScope 嵌入模型: {EMBEDDING_MODEL}")
        
    elif provider == "openai":
        # OpenAI 嵌入模型
        from langchain_openai import OpenAIEmbeddings
        if not OPENAI_API_KEY:
            raise ValueError("使用 OpenAI 嵌入模型需要设置 OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        print(f"使用 OpenAI 嵌入模型: {EMBEDDING_MODEL}")
        
    elif provider == "huggingface":
        # HuggingFace 本地嵌入模型
        from langchain_community.embeddings import HuggingFaceEmbeddings
        # 如果指定了完整路径则使用，否则自动补全 sentence-transformers/ 前缀
        if HUGGINGFACE_MODEL_PATH:
            model_name = HUGGINGFACE_MODEL_PATH
        elif EMBEDDING_MODEL.startswith("sentence-transformers/") or "/" in EMBEDDING_MODEL:
            model_name = EMBEDDING_MODEL
        else:
            # 自动补全 sentence-transformers/ 前缀
            model_name = f"sentence-transformers/{EMBEDDING_MODEL}"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"使用 HuggingFace 嵌入模型: {model_name}")
        
    elif provider == "ollama":
        # Ollama 本地嵌入模型
        from langchain_community.embeddings import OllamaEmbeddings
        embeddings = OllamaEmbeddings(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        print(f"使用 Ollama 嵌入模型: {OLLAMA_MODEL} (地址: {OLLAMA_BASE_URL})")
        
    else:
        raise ValueError(f"不支持的嵌入模型提供商: {provider}。支持: dashscope, openai, huggingface, ollama")
    
    return embeddings

def test_embedding_model(embeddings):
    """
    入参:
    - embeddings: 嵌入模型实例
    方法:
    - 测试嵌入模型是否正常工作
    - 通过嵌入一个简单文本验证 API 连接
    出参:
    - bool: 测试是否成功
    """
    print("正在测试嵌入模型连接...")
    test_text = "测试"
    test_result = embeddings.embed_query(test_text)
    if test_result and len(test_result) > 0:
        print(f"嵌入模型测试成功，向量维度: {len(test_result)}")
        return True
    else:
        print("嵌入模型测试失败：返回结果为空")
        return False

def create_vector_store(documents=None, force_recreate=False, interactive=True):
    """
    入参:
    - documents (list[Document], optional): 要向量化的文档列表，如果为 None 则从文件加载
    - force_recreate (bool): 是否强制重新创建向量数据库
    方法:
    - 初始化 DashScope 嵌入模型
    - 测试嵌入模型连接是否正常
    - 如果向量数据库不存在或强制重建，则创建新的向量存储
    - 将文档块向量化并存储到 ChromaDB
    - 如果向量数据库已存在，则直接加载
    出参:
    - vectorstore (Chroma): 向量数据库实例
    """
    # 初始化嵌入模型
    print("正在初始化嵌入模型...")
    embeddings = create_embeddings()
    
    # 测试嵌入模型连接
    if not test_embedding_model(embeddings):
        raise ConnectionError("嵌入模型连接失败，请检查 API Key 和网络连接")
    
    # 如果强制重建，先删除旧的向量数据库目录
    if force_recreate and os.path.exists(VECTOR_DB_DIR):
        print("检测到强制重建标志，正在删除旧的向量数据库...")
        shutil.rmtree(VECTOR_DB_DIR)
        print("旧向量数据库已删除")
    
    # 检查向量数据库是否存在
    if os.path.exists(VECTOR_DB_DIR) and not force_recreate:
        print("加载已存在的向量数据库...")
        # 验证向量数据库是否有效
        vectorstore = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings
        )
        # 尝试检索以验证数据库有效性
        try:
            collection_count = vectorstore._collection.count()
            print(f"向量数据库加载完成，包含 {collection_count} 个文档块")
        except Exception as e:
            print(f"警告：向量数据库可能已损坏，将重新创建。错误: {e}")
            force_recreate = True
            # 删除损坏的数据库
            if os.path.exists(VECTOR_DB_DIR):
                shutil.rmtree(VECTOR_DB_DIR)
        
        if not force_recreate:
            return vectorstore
    
    # 创建新的向量数据库
    print("创建新的向量数据库...")
    
    # 如果没有提供文档，则加载文档
    if documents is None:
        documents = load_documents()
    
    if not documents:
        raise ValueError("没有找到可用的文档，请检查知识库目录")
    
    # 分割文档
    splits = split_documents(documents)
    
    if not splits:
        raise ValueError("文档分割后为空，请检查文档内容")
    
    print(f"开始向量化 {len(splits)} 个文档块，这可能需要一些时间...")
    print(f"批量处理配置：每批 {BATCH_SIZE} 个文档块，批次延迟 {BATCH_DELAY} 秒")
    
    # 批量处理文档块，避免连接中断
    import time
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    
    @retry(
        stop=stop_after_attempt(RETRY_TIMES),
        wait=wait_exponential(multiplier=1, min=RETRY_DELAY, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def batch_add_documents(vectorstore, batch_docs, batch_num, total_batches):
        """批量添加文档到向量数据库，带重试机制"""
        try:
            print(f"  处理批次 {batch_num}/{total_batches} ({len(batch_docs)} 个文档块)...", end="", flush=True)
            vectorstore.add_documents(batch_docs)
            print(" ✅")
            return True
        except Exception as e:
            print(f" ❌ 错误: {e}")
            raise
    
    # 创建空的向量存储
    print("创建向量数据库...")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    
    # 分批处理文档块
    total_batches = (len(splits) + BATCH_SIZE - 1) // BATCH_SIZE
    successful_batches = 0
    failed_batches = []
    
    for i in range(0, len(splits), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch_docs = splits[i:i + BATCH_SIZE]
        
        try:
            batch_add_documents(vectorstore, batch_docs, batch_num, total_batches)
            successful_batches += 1
            
            # 批次之间的延迟，避免请求过快
            if i + BATCH_SIZE < len(splits):
                time.sleep(BATCH_DELAY)
                
        except Exception as e:
            print(f"\n⚠️  批次 {batch_num} 处理失败，已重试 {RETRY_TIMES} 次")
            print(f"   错误信息: {e}")
            failed_batches.append((batch_num, batch_docs, str(e)))
            
            # 询问是否继续（仅在交互模式下）
            if interactive:
                print(f"   是否跳过此批次继续处理？(y/n): ", end="", flush=True)
                try:
                    choice = input().strip().lower()
                    if choice != 'y':
                        print("\n❌ 用户取消操作")
                        raise Exception("用户取消向量化操作")
                except KeyboardInterrupt:
                    print("\n\n❌ 操作被中断")
                    raise
            else:
                print(f"   自动跳过此批次，继续处理...")
    
    # 注意：新版本 ChromaDB 使用 persist_directory 参数时会自动持久化，无需手动调用 persist()
    # vectorstore.persist()  # 此方法在新版本中已被移除
    
    # 统计结果
    print(f"\n向量化完成统计：")
    print(f"  总文档块数: {len(splits)}")
    print(f"  成功批次: {successful_batches}/{total_batches}")
    if failed_batches:
        print(f"  失败批次: {len(failed_batches)}")
        print(f"  失败批次编号: {[b[0] for b in failed_batches]}")
    
    if failed_batches:
        print(f"\n⚠️  警告：有 {len(failed_batches)} 个批次处理失败")
        print("   可以稍后重新运行向量化脚本，已成功的批次不会重复处理")
    
    # 验证向量数据库
    try:
        collection_count = vectorstore._collection.count()
        print(f"\n✅ 向量数据库创建完成，共存储 {collection_count} 个文档块")
    except Exception as e:
        print(f"\n⚠️  无法验证向量数据库: {e}")
    
    return vectorstore

def rerank_with_cross_encoder(docs, query, top_k):
    """
    入参:
    - docs (list[Document]): 向量检索返回的候选文档列表
    - query (str): 用户原始查询文本
    - top_k (int): 重排序后需要返回的文档数量
    方法:
    - 调用 DashScope TextReRank Cross-Encoder 模型（gte-rerank-v2）
    - 将 query 与每个 doc 的文本拼接，通过 Cross-Encoder 计算精确语义相关性分数
    - Cross-Encoder 比 Bi-Encoder（向量检索）更准确，因为它在注意力层同时编码 query 和 doc
    - 按相关性分数降序排列，返回 top_k 个文档
    出参:
    - reranked_docs (list[Document]): 按 Cross-Encoder 分数排序后的文档列表（长度 ≤ top_k）
    """
    if not docs or len(docs) == 0:
        return docs

    if len(docs) <= top_k:
        return docs

    # 截断文档文本，Cross-Encoder 对单条文档有 token 上限（约 500 token）
    # 保留前 800 字符，在覆盖核心语义的同时避免超出 API 限制
    doc_texts = [doc.page_content[:800] for doc in docs]

    # 调用 DashScope Cross-Encoder 重排 API
    # 显式传入 api_key，避免 SDK 无法从环境变量自动获取
    import dashscope
    from dashscope import TextReRank
    dashscope.api_key = QWEN_API_KEY
    response = TextReRank.call(
        model="gte-rerank-v2",
        query=query,
        documents=doc_texts
    )

    if response.status_code != 200:
        print(f"⚠️  Cross-Encoder 重排失败 (status={response.status_code}, message={response.message})，回退到原始排序")
        return docs[:top_k]

    # response.output["results"] 已按 relevance_score 降序排列
    # 每个 result 包含 index（原文档索引）和 relevance_score（相关性分数）
    ranked_results = response.output.get("results", [])

    # 按 Cross-Encoder 返回的顺序重建文档列表，取 top_k
    reranked_docs = []
    for result in ranked_results[:top_k]:
        idx = result["index"]
        score = result["relevance_score"]
        doc = docs[idx]
        # 将 Cross-Encoder 分数写入 metadata，供后续展示使用
        doc.metadata["relevance_score"] = score
        reranked_docs.append(doc)

    return reranked_docs

def get_retriever(vectorstore, k=4, score_threshold=None, use_rerank=False):
    """
    入参:
    - vectorstore (Chroma): 向量数据库实例
    - k (int): 最终返回的文档数量（重排序后的 top_k）
    - score_threshold (float, optional): 相似度阈值，低于此值的文档将被过滤
    - use_rerank (bool): 是否启用 Cross-Encoder 重排序
    方法:
    - 创建基础向量检索器（Bi-Encoder 粗筛）
    - 如果启用重排序，先用大 K 粗筛，再用 DashScope Cross-Encoder 精排
    - 未启用重排序时直接返回向量检索结果
    出参:
    - retriever: 检索器实例（可能包含 Cross-Encoder 重排序包装）
    """
    # 基础检索器配置：重排序模式先多取候选文档用于后续精排
    if use_rerank:
        initial_k = INITIAL_RETRIEVE_K
        search_kwargs = {"k": initial_k}
    else:
        search_kwargs = {"k": k}

    if score_threshold:
        search_kwargs["score_threshold"] = score_threshold

    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )

    # 启用重排序时，包装基础检索器，在检索后调用 Cross-Encoder 精排
    if use_rerank:
        class CrossEncoderRetriever:
            """包装检索器：向量粗筛 → Cross-Encoder 精排"""
            def __init__(self, base_retriever, top_k):
                self.base_retriever = base_retriever
                self.top_k = top_k

            def invoke(self, query):
                # 第一步：向量检索粗筛（Bi-Encoder，快但不够精确）
                docs = self.base_retriever.invoke(query)
                # 第二步：Cross-Encoder 精排（慢但精确）
                return rerank_with_cross_encoder(docs, query, self.top_k)

        retriever = CrossEncoderRetriever(base_retriever, k)
        print(f"已启用 Cross-Encoder 重排序（初始检索 {initial_k} 个，精排后返回 {k} 个）")
    else:
        retriever = base_retriever

    return retriever

def search_knowledge_base(vectorstore, query, k=4, use_rerank=False):
    """
    入参:
    - vectorstore (Chroma): 向量数据库实例
    - query (str): 查询文本
    - k (int): 返回的相似文档数量
    - use_rerank (bool): 是否使用 Cross-Encoder 重排序
    方法:
    - 在知识库中搜索与查询最相关的文档块
    - 使用向量相似度搜索找到最匹配的内容
    - 如果启用重排序，使用 Cross-Encoder 对结果进行精排
    出参:
    - results (list[Document]): 检索到的相关文档列表
    """
    retriever = get_retriever(vectorstore, k=k, use_rerank=use_rerank)
    results = retriever.invoke(query)
    return results

