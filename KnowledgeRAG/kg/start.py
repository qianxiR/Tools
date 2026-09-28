"""
知识图谱 AI Agent 统一启动脚本

入参:
- 无
方法:
- 显示启动菜单，提供三种启动方式选择
- 根据用户选择启动对应的 Agent 模块
出参:
- 无
"""
import sys

def show_menu():
    """
    入参:
    - 无
    方法:
    - 显示启动菜单选项
    出参:
    - 无
    """
    print("=" * 60)
    print("🤖 知识图谱 AI Agent 启动菜单")
    print("=" * 60)
    print("1. 连接测试（base.py）- 快速验证 Neo4j 连接")
    print("2. 基础 Agent 示例（kg_agent.py）- 运行预设查询示例")
    print("3. 智能对话 Agent（langchain_agent.py）- 自然语言对话")
    print("0. 退出")
    print("=" * 60)

def main():
    """
    入参:
    - 无
    方法:
    - 主函数，循环显示菜单并处理用户选择
    - 根据选择启动对应的模块
    出参:
    - 无
    """
    while True:
        show_menu()
        choice = input("\n请选择启动方式 (0-3): ").strip()
        
        if choice == "0":
            print("👋 再见！")
            break
        elif choice == "1":
            print("\n" + "=" * 60)
            print("正在启动连接测试...")
            print("=" * 60 + "\n")
            try:
                import base
            except Exception as e:
                print(f"❌ 启动失败: {e}")
        elif choice == "2":
            print("\n" + "=" * 60)
            print("正在启动基础 Agent 示例...")
            print("=" * 60 + "\n")
            try:
                import kg_agent
            except Exception as e:
                print(f"❌ 启动失败: {e}")
        elif choice == "3":
            print("\n" + "=" * 60)
            print("正在启动智能对话 Agent...")
            print("=" * 60 + "\n")
            try:
                import langchain_agent
            except Exception as e:
                print(f"❌ 启动失败: {e}")
        else:
            print("❌ 无效选择，请重新输入")
        
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()

