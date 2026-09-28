"""
基于 LangChain 1.0 API 的 Qwen 聊天实现
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 全局变量配置
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-turbo"
TEMPERATURE = 0.7
MAX_TOKENS = 2000

def init_chat_model():
    """
    入参:
    - 无
    方法:
    - 初始化 Qwen 聊天模型，配置 API key、base_url 和模型参数
    - 使用 LangChain 1.0 的 ChatOpenAI 类，通过 OpenAI 兼容接口调用 Qwen
    出参:
    - llm (ChatOpenAI): 初始化完成的聊天模型实例
    """
    llm = ChatOpenAI(
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
        model=QWEN_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    return llm

def chat_once(llm, user_input, system_prompt=None):
    """
    入参:
    - llm (ChatOpenAI): 已初始化的聊天模型实例
    - user_input (str): 用户输入的聊天内容
    - system_prompt (str, optional): 系统提示词，用于设定助手角色和行为
    方法:
    - 构建消息列表，包含系统消息（可选）和用户消息
    - 调用模型的 invoke 方法获取回复
    - 返回助手的回复内容
    出参:
    - response (str): 模型生成的回复文本
    """
    messages = []
    
    # 添加系统提示词（如果提供）
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    
    # 添加用户消息
    messages.append(HumanMessage(content=user_input))
    
    # 调用模型获取回复
    response = llm.invoke(messages)
    
    return response.content

def chat_stream(llm, user_input, system_prompt=None):
    """
    入参:
    - llm (ChatOpenAI): 已初始化的聊天模型实例
    - user_input (str): 用户输入的聊天内容
    - system_prompt (str, optional): 系统提示词，用于设定助手角色和行为
    方法:
    - 构建消息列表，包含系统消息（可选）和用户消息
    - 使用 stream 方法进行流式输出，逐块返回生成内容
    - 适用于需要实时显示回复的场景
    出参:
    - generator: 生成器对象，每次 yield 返回一个文本块
    """
    messages = []
    
    # 添加系统提示词（如果提供）
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    
    # 添加用户消息
    messages.append(HumanMessage(content=user_input))
    
    # 流式调用模型
    for chunk in llm.stream(messages):
        yield chunk.content

def main():
    """
    入参:
    - 无
    方法:
    - 主函数，演示如何使用 Qwen 聊天功能
    - 初始化模型，进行单次对话和流式对话示例
    出参:
    - 无
    """
    # 初始化聊天模型
    print("正在初始化 Qwen 聊天模型...")
    llm = init_chat_model()
    print("模型初始化完成！\n")
    
    # 示例1: 单次对话
    print("=" * 50)
    print("示例1: 单次对话")
    print("=" * 50)
    user_question = "请简单介绍一下 LangChain"
    system_prompt = "你是一个专业的 AI 助手，擅长用简洁清晰的语言回答问题。"
    
    response = chat_once(llm, user_question, system_prompt)
    print(f"用户: {user_question}")
    print(f"助手: {response}\n")
    
    # 示例2: 流式对话
    print("=" * 50)
    print("示例2: 流式对话")
    print("=" * 50)
    user_question2 = "Python 中的装饰器是什么？"
    print(f"用户: {user_question2}")
    print("助手: ", end="", flush=True)
    
    for chunk in chat_stream(llm, user_question2):
        print(chunk, end="", flush=True)
    print("\n")
    
    # 示例3: 交互式聊天循环
    print("=" * 50)
    print("示例3: 交互式聊天（输入 'quit' 或 'exit' 退出）")
    print("=" * 50)
    
    while True:
        user_input = input("\n你: ").strip()
        
        if user_input.lower() in ['quit', 'exit', '退出']:
            print("再见！")
            break
        
        if not user_input:
            continue
        
        print("助手: ", end="", flush=True)
        for chunk in chat_stream(llm, user_input):
            print(chunk, end="", flush=True)
        print()

if __name__ == "__main__":
    main()

