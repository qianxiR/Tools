"""
RAG Agent 模块 - 集成知识库检索的智能助手（支持重排序和流式对话）
"""
import os
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from utils.embeding import create_vector_store, get_retriever, USE_RERANK, RERANK_TOP_K, INITIAL_RETRIEVE_K

# 全局变量配置
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
QWEN_MODEL = "qwen-plus"
TEMPERATURE = 0.7
MAX_TOKENS = 2000

def format_docs(docs):
    """
    入参:
    - docs (list[Document]): 文档列表
    方法:
    - 将检索到的文档格式化为字符串
    - 包含文档来源信息，便于追溯
    - 支持重排序后的文档（可能包含相关性分数）
    出参:
    - formatted_str (str): 格式化后的文档内容字符串
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        content = doc.page_content
        
        # 如果有相关性分数，显示分数
        if hasattr(doc, 'metadata') and 'relevance_score' in doc.metadata:
            score = doc.metadata['relevance_score']
            formatted.append(f"[文档 {i}] 来源: {source} (相关性: {score:.3f})\n{content}\n")
        else:
            formatted.append(f"[文档 {i}] 来源: {source}\n{content}\n")
    
    return "\n---\n".join(formatted)

def create_rag_chain(vectorstore, use_rerank=None):
    """
    入参:
    - vectorstore: 向量数据库实例
    - use_rerank (bool, optional): 是否使用重排序，默认使用全局配置
    方法:
    - 创建 RAG 检索链，集成知识库检索和 LLM 生成
    - 如果启用重排序，使用 Cross-Encoder 对检索结果进行精排
    - 构建提示词模板，指导模型使用检索到的知识回答问题
    - 配置检索器，从知识库中获取相关上下文
    出参:
    - chain: RAG 链实例
    """
    # 初始化 LLM（仅用于生成回答）
    llm = ChatTongyi(
        model=QWEN_MODEL,
        dashscope_api_key=QWEN_API_KEY,
        temperature=TEMPERATURE
    )
    
    # 确定是否使用重排序
    if use_rerank is None:
        use_rerank = USE_RERANK
    
    # 创建检索器（如果启用重排序，会使用 Cross-Encoder 进行精排）
    retriever = get_retriever(
        vectorstore,
        k=RERANK_TOP_K if use_rerank else 4,
        use_rerank=use_rerank
    )

    # 构建提示词模板
    template = """你是一个专业的 AI 助手，擅长根据提供的知识库内容回答问题。

请根据以下知识库内容回答用户的问题。如果知识库中没有相关信息，请基于你的通用知识回答，但需要说明信息来源。

知识库内容：
{context}

用户问题：{question}

请提供准确、详细的回答："""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 构建 RAG 链
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def create_agent_with_memory(vectorstore, use_rerank=None):
    """
    入参:
    - vectorstore: 向量数据库实例
    - use_rerank (bool, optional): 是否使用重排序，默认使用全局配置
    方法:
    - 创建带对话记忆的 Agent
    - 支持多轮对话，保留上下文
    - 集成知识库检索功能（支持 Cross-Encoder 重排序）
    - 支持流式输出
    出参:
    - agent: Agent 实例（包含 chat 和 chat_stream 方法）
    """
    # 初始化 LLM
    llm = ChatTongyi(
        model=QWEN_MODEL,
        dashscope_api_key=QWEN_API_KEY,
        temperature=TEMPERATURE
    )
    
    # 确定是否使用重排序
    if use_rerank is None:
        use_rerank = USE_RERANK
    
    # 创建检索器（如果启用重排序，会使用 Cross-Encoder 进行精排）
    retriever = get_retriever(
        vectorstore,
        k=RERANK_TOP_K if use_rerank else 4,
        use_rerank=use_rerank
    )

    # 对话历史
    conversation_history = []
    # 最后一次检索到的文档（用于显示来源）
    last_retrieved_docs = []
    
    def should_retrieve_knowledge(user_input):
        """
        入参:
        - user_input (str): 用户输入
        方法:
        - 判断是否需要检索知识库
        - 使用 LLM 快速分析问题，判断是否需要专业知识库支持
        - 如果是专业问题（涉及案例、工具、技术细节等），返回 True
        - 如果是通用问题或闲聊，返回 False
        出参:
        - need_retrieve (bool): 是否需要检索知识库
        """
        # 构建判断消息
        judge_messages = [
            SystemMessage(content="""你是一个智能助手，需要判断用户的问题是否需要检索专业知识库。

知识库包含以下内容：
- 解决方案案例（solution_cases）：植被恢复监测、土壤恢复评估、状态演化分析、多尺度监测等案例
- 知识库（knowledge）：实体、指数、状态、时间尺度等专业知识
- 工具（utils）：NDVI计算、NDWI计算、NBR计算、坡度面积计算、多时序分析等工具

判断标准：
1. 如果问题涉及专业知识、具体案例、工具使用、技术细节等，需要检索知识库
2. 如果问题是一般性对话、闲聊、通用知识问答等，不需要检索知识库
3. 如果不确定，倾向于检索知识库以确保准确性

请只回答"需要"或"不需要"，不要添加其他内容。"""),
            HumanMessage(content=f"用户问题：{user_input}\n\n是否需要检索知识库？")
        ]
        
        # 调用 LLM 判断
        try:
            response = llm.invoke(judge_messages)
            judge_result = response.content.strip().lower()
            
            # 判断结果
            need_retrieve = "需要" in judge_result or "yes" in judge_result or "true" in judge_result
            return need_retrieve
        except Exception as e:
            # 如果判断失败，默认检索知识库以确保准确性
            print(f"⚠️  判断是否需要检索知识库时出错: {e}，默认检索知识库")
            return True
    
    def chat(user_input, use_rag="auto"):
        """
        入参:
        - user_input (str): 用户输入
        - use_rag (str/bool): RAG 使用模式
            - "auto": 自动判断是否需要检索（默认）
            - True: 强制使用 RAG 检索
            - False: 不使用 RAG 检索
        方法:
        - 处理用户输入，根据模式决定是否检索知识库
        - 如果 use_rag="auto"，先判断是否需要检索
        - 构建包含上下文的提示词
        - 调用 LLM 生成回复（非流式）
        - 保存对话历史
        出参:
        - response (str): AI 回复内容
        """
        # 确定是否需要检索知识库
        if use_rag == "auto":
            # 自动判断是否需要检索
            need_retrieve = should_retrieve_knowledge(user_input)
        elif use_rag is True:
            need_retrieve = True
        else:
            need_retrieve = False
        
        # 检索相关知识库内容
        if need_retrieve:
            relevant_docs = retriever.invoke(user_input)
            context = format_docs(relevant_docs)
            # 保存检索到的文档（用于显示来源）
            last_retrieved_docs.clear()
            last_retrieved_docs.extend(relevant_docs)
        else:
            context = ""
            last_retrieved_docs.clear()
        
        # 构建消息列表
        messages = []
        
        # 添加系统提示
        if context:
            # 有知识库内容时的提示
            system_content = """你是一个专业的 AI 助手，擅长根据知识库内容回答问题。

请根据提供的知识库内容回答用户的问题。如果知识库中没有相关信息，请基于你的通用知识回答，但需要说明信息来源。

保持对话的自然流畅，参考历史对话上下文。

相关知识库内容：
{context}""".format(context=context)
        else:
            # 没有知识库内容时的提示（通用对话）
            system_content = """你是一个专业的 AI 助手，擅长回答各种问题。

请基于你的通用知识回答用户的问题，保持对话的自然流畅，参考历史对话上下文。"""
        
        messages.append(SystemMessage(content=system_content))
        
        # 添加对话历史（最近 3 轮）
        for msg in conversation_history[-6:]:  # 保留最近 3 轮对话（每轮 2 条消息）
            messages.append(msg)
        
        # 添加当前用户输入
        messages.append(HumanMessage(content=user_input))
        
        # 调用 LLM（非流式）
        response = llm.invoke(messages)
        response_content = response.content
        
        # 保存对话历史
        conversation_history.append(HumanMessage(content=user_input))
        conversation_history.append(AIMessage(content=response_content))
        
        # 限制历史长度，避免超出 token 限制
        if len(conversation_history) > 20:
            conversation_history.pop(0)
            conversation_history.pop(0)
        
        return response_content
    
    def chat_stream(user_input, use_rag="auto"):
        """
        入参:
        - user_input (str): 用户输入
        - use_rag (str/bool): RAG 使用模式
            - "auto": 自动判断是否需要检索（默认）
            - True: 强制使用 RAG 检索
            - False: 不使用 RAG 检索
        方法:
        - 处理用户输入，根据模式决定是否检索知识库
        - 如果 use_rag="auto"，先判断是否需要检索
        - 构建包含上下文的提示词
        - 流式调用 LLM 生成回复
        - 保存对话历史
        出参:
        - generator: 生成器对象，每次 yield 返回一个文本块
        """
        # 确定是否需要检索知识库
        if use_rag == "auto":
            # 自动判断是否需要检索
            need_retrieve = should_retrieve_knowledge(user_input)
        elif use_rag is True:
            need_retrieve = True
        else:
            need_retrieve = False
        
        # 检索相关知识库内容
        if need_retrieve:
            relevant_docs = retriever.invoke(user_input)
            context = format_docs(relevant_docs)
            # 保存检索到的文档（用于显示来源）
            last_retrieved_docs.clear()
            last_retrieved_docs.extend(relevant_docs)
        else:
            context = ""
            last_retrieved_docs.clear()
        
        # 构建消息列表
        messages = []
        
        # 添加系统提示
        if context:
            # 有知识库内容时的提示
            system_content = """你是一个专业的 AI 助手，擅长根据知识库内容回答问题。

请根据提供的知识库内容回答用户的问题。如果知识库中没有相关信息，请基于你的通用知识回答，但需要说明信息来源。

保持对话的自然流畅，参考历史对话上下文。

相关知识库内容：
{context}""".format(context=context)
        else:
            # 没有知识库内容时的提示（通用对话）
            system_content = """你是一个专业的 AI 助手，擅长回答各种问题。

请基于你的通用知识回答用户的问题，保持对话的自然流畅，参考历史对话上下文。"""
        
        messages.append(SystemMessage(content=system_content))
        
        # 添加对话历史（最近 3 轮）
        for msg in conversation_history[-6:]:  # 保留最近 3 轮对话（每轮 2 条消息）
            messages.append(msg)
        
        # 添加当前用户输入
        messages.append(HumanMessage(content=user_input))
        
        # 流式调用 LLM
        full_response = ""
        for chunk in llm.stream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                full_response += chunk.content
                yield chunk.content
        
        # 保存对话历史
        conversation_history.append(HumanMessage(content=user_input))
        conversation_history.append(AIMessage(content=full_response))
        
        # 限制历史长度，避免超出 token 限制
        if len(conversation_history) > 20:
            conversation_history.pop(0)
            conversation_history.pop(0)
    
    def clear_history():
        """清空对话历史"""
        conversation_history.clear()
    
    # 创建 Agent 类，避免 self 参数问题
    class Agent:
        def chat(self, user_input, use_rag="auto"):
            return chat(user_input, use_rag)
        
        def chat_stream(self, user_input, use_rag="auto"):
            return chat_stream(user_input, use_rag)
        
        def clear_history(self):
            clear_history()
        
        @property
        def history(self):
            return conversation_history
        
        @property
        def last_retrieved_docs(self):
            """获取最后一次检索到的文档列表"""
            return last_retrieved_docs.copy()
    
    agent = Agent()
    
    return agent

def create_simple_rag_agent(vectorstore, use_rerank=None):
    """
    入参:
    - vectorstore: 向量数据库实例
    - use_rerank (bool, optional): 是否使用重排序，默认使用全局配置
    方法:
    - 创建简单的 RAG Agent，使用链式调用
    - 适合单次问答场景
    - 支持重排序功能
    出参:
    - rag_chain: RAG 链实例
    """
    return create_rag_chain(vectorstore, use_rerank=use_rerank)

