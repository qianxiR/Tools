"""
Google Earth Engine Python API 认证脚本

入参:
- 无（交互式认证）

方法:
- 检查是否已存在有效的认证凭据
- 如果未认证，引导用户完成认证流程
- 验证认证是否成功

出参:
- 认证凭据保存到本地配置文件
- 输出认证状态信息
"""

import os
import sys

def check_credentials_exist():
    """
    检查认证凭据文件是否存在
    
    入参:
    - 无
    
    方法:
    - 检查标准的 GEE 凭据文件路径
    - 支持 Windows、Linux、Mac 三种操作系统
    
    出参:
    - bool: 凭据文件是否存在
    """
    # 获取用户主目录
    home = os.path.expanduser("~")
    
    # GEE 凭据文件路径
    credentials_path = os.path.join(home, ".config", "earthengine", "credentials")
    
    return os.path.exists(credentials_path)


def authenticate_gee():
    """
    执行 GEE 认证流程
    
    入参:
    - 无
    
    方法:
    - 引导用户完成浏览器认证
    - 如果浏览器未自动打开，提供手动URL
    - 保存认证凭据到本地
    
    出参:
    - bool: 认证是否成功
    """
    # 动态导入 ee 模块
    import ee
    
    print("\n==================== GEE Python API 认证 ====================")
    print("即将打开浏览器进行认证...")
    print("请按照以下步骤操作：")
    print("1. 在浏览器中登录 Google 账号（使用已注册 GEE 的账号）")
    print("2. 授权应用访问 Google Earth Engine")
    print("3. 复制获得的验证码")
    print("4. 将验证码粘贴到终端中")
    print("\n⚠️  如果浏览器未自动打开，请手动复制下方的认证URL到浏览器")
    print("===========================================================\n")
    
    # 执行认证（force=True 强制生成新的认证URL）
    ee.Authenticate(force=False)
    
    # 验证认证是否成功
    if check_credentials_exist():
        print("\n✅ 认证成功！凭据已保存到本地")
        return True
    else:
        print("\n❌ 认证失败，请重试")
        return False


def get_project_id():
    """
    获取或输入 GEE Cloud Project ID
    
    入参:
    - 无
    
    方法:
    - 首先尝试从环境变量读取
    - 如果不存在，提示用户输入
    - 保存到配置文件以便下次使用
    
    出参:
    - str: Cloud Project ID
    """
    # 默认项目ID（根据您的项目）
    default_project = "applied-pipe-453411-k9"
    
    # 配置文件路径
    home = os.path.expanduser("~")
    config_dir = os.path.join(home, ".config", "earthengine")
    config_file = os.path.join(config_dir, "project_id.txt")
    
    # 尝试从配置文件读取
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                saved_project = f.read().strip()
                if saved_project:
                    print(f"✓ 使用已保存的 Project ID: {saved_project}")
                    return saved_project
        except UnicodeDecodeError:
            print("⚠️  项目ID文件编码有问题，将重新创建")
            # 删除有问题的文件
            os.remove(config_file)
    
    # 提示用户输入
    print("\n⚠️  GEE Python API 需要 Cloud Project ID")
    print("您可以在 Google Cloud Console 中找到您的项目ID")
    print("访问：https://console.cloud.google.com/")
    
    project_input = input(f"\n请输入您的 Project ID（直接回车使用默认：{default_project}）: ").strip()
    
    project_id = project_input if project_input else default_project
    
    # 保存到配置文件
    os.makedirs(config_dir, exist_ok=True)
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(project_id)
    
    print(f"✓ Project ID 已保存: {project_id}")
    return project_id


def test_connection(project_id=None):
    """
    测试 GEE 连接是否正常
    
    入参:
    - project_id (str): Cloud Project ID，如果为None则自动获取
    
    方法:
    - 使用指定的 project_id 初始化 GEE
    - 执行简单的 API 调用验证连接
    
    出参:
    - bool: 连接是否正常
    """
    import ee
    
    # 获取 project_id
    if project_id is None:
        project_id = get_project_id()
    
    # 初始化 GEE（新版API必须指定project）
    print(f"\n正在初始化 GEE (Project: {project_id})...")
    ee.Initialize(project=project_id)
    print("✓ GEE 初始化成功")
    
    print("\n==================== 测试 GEE 连接 ====================")
    
    # 测试 1: 获取项目信息
    print("测试 1: 验证项目信息...")
    print(f"✓ Cloud Project: {project_id}")
    
    # 测试 2: 访问数据集
    print("\n测试 2: 访问 Sentinel-2 数据集...")
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    count = s2.filterDate('2024-01-01', '2024-01-31').limit(100).size().getInfo()
    print(f"✓ 成功访问数据集，2024年1月影像数量: {count}")
    
    # 测试 3: 执行简单计算
    print("\n测试 3: 执行简单计算...")
    dem = ee.Image('NASA/NASADEM_HGT/001')
    mean_elevation = dem.select('elevation').reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=ee.Geometry.Point([116.4, 39.9]),  # 北京坐标
        scale=30
    ).getInfo()
    print(f"✓ 北京地区平均海拔: {mean_elevation['elevation']:.2f} 米")
    
    print("\n====================================================")
    print("✅ 所有测试通过，GEE 连接正常！")
    
    return True


def main():
    """
    主函数：GEE 认证流程控制
    
    入参:
    - 无
    
    方法:
    - 检查是否已认证
    - 未认证则引导用户完成认证
    - 测试连接是否正常
    
    出参:
    - 无
    """
    print("==================== GEE Python API 认证工具 ====================")
    print("此脚本将帮助您完成 Google Earth Engine Python API 的认证配置")
    print("==============================================================\n")
    
    # 步骤 1: 检查 earthengine-api 是否已安装
    print("步骤 1: 检查依赖包...")
    ee_installed = False
    
    try:
        import ee
        print("✓ earthengine-api 已安装")
        print(f"  版本: {ee.__version__}")
        ee_installed = True
    except ImportError:
        print("❌ earthengine-api 未安装")
        print("\n请先安装 earthengine-api：")
        print("  pip install earthengine-api")
        print("\n国内用户推荐使用清华镜像源：")
        print("  pip install earthengine-api -i https://pypi.tuna.tsinghua.edu.cn/simple")
        return
    
    # 步骤 2: 检查是否已认证
    print("\n步骤 2: 检查认证状态...")
    if check_credentials_exist():
        print("✓ 检测到已存在的认证凭据")
        
        # 尝试测试连接
        choice = input("\n是否测试当前认证是否有效？(y/n): ").strip().lower()
        if choice == 'y':
            try:
                test_connection()
                print("\n当前认证有效，无需重新认证")
                return
            except Exception as e:
                print(f"\n❌ 认证测试失败: {str(e)}")
                print("可能需要重新认证")
        
        # 询问是否重新认证
        choice = input("\n是否重新认证？(y/n): ").strip().lower()
        if choice != 'y':
            print("退出认证流程")
            return
    else:
        print("✗ 未检测到认证凭据")
    
    # 步骤 3: 执行认证
    print("\n步骤 3: 开始认证流程...")
    print("\n⚠️  注意事项：")
    print("1. 确保您已注册 GEE 账号并通过审核")
    print("2. 认证过程需要访问 Google 服务（可能需要特殊网络环境）")
    print("3. 请准备好您的 Google 账号密码")
    
    choice = input("\n准备好了吗？按 Enter 继续，或输入 'q' 退出: ").strip().lower()
    if choice == 'q':
        print("退出认证流程")
        return
    
    # 执行认证
    success = authenticate_gee()
    
    if not success:
        print("\n认证失败，请检查网络连接或重试")
        return
    
    # 步骤 4: 配置 Cloud Project
    print("\n步骤 4: 配置 Cloud Project ID...")
    project_id = get_project_id()
    
    # 步骤 5: 测试连接
    print("\n步骤 5: 测试连接...")
    try:
        test_connection(project_id)
        print("\n🎉 认证配置完成！您现在可以使用 GEE Python API 了")
        print("\n后续使用方法：")
        print("```python")
        print("import ee")
        print(f"ee.Initialize(project='{project_id}')  # 初始化连接")
        print("# 开始使用 GEE API...")
        print("```")
    except Exception as e:
        print(f"\n❌ 连接测试失败: {str(e)}")
        print("请检查认证是否成功或稍后重试")


if __name__ == "__main__":
    main()

