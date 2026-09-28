"""
完整的湖库水深数据提取与预处理流程
功能：
1. 可选：检查Sentinel-2影像覆盖情况
2. 提取Sentinel-2波段值（调用extract_sentinel_band_values.py）
3. 合并批次文件并进行预处理（调用merge_and_preprocess.py）
4. 生成完整的数据管道和报告
"""

import subprocess
import sys
import os
import time
import glob
from pathlib import Path
import locale

# 获取系统默认编码
SYSTEM_ENCODING = locale.getpreferredencoding() or 'utf-8'

# 导入配置文件
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *

# ========== 配置参数 ==========
# 使用配置文件中的参数
WORK_DIR = Path(BASE_DIR)
CODE_DIR = Path(CODE_DIR)
SCRIPT_EXTRACT = CODE_DIR / 'extract_sentinel_band_values.py'
SCRIPT_MERGE = CODE_DIR / 'merge_and_preprocess.py'
SCRIPT_CHECK = CODE_DIR / 'check_image_coverage_only.py'

# 步骤控制
RUN_COVERAGE_CHECK = ENABLE_COVERAGE_CHECK
RUN_EXTRACTION = ENABLE_EXTRACTION
RUN_MERGE_PREPROCESS = ENABLE_MERGE_PREPROCESS

# 输出文件
OUTPUT_EXTRACT = WORK_DIR / OUTPUT_EXTRACT_CSV
OUTPUT_MERGED = WORK_DIR / OUTPUT_MERGED_CSV
OUTPUT_PROCESSED = WORK_DIR / OUTPUT_PROCESSED_CSV
OUTPUT_REPORT = WORK_DIR / OUTPUT_REPORT_TXT
TEMP_PATTERN = TEMP_BATCH_PATTERN


def print_header(title):
    """打印标题"""
    print('\n' + '=' * 80)
    print(title)
    print('=' * 80)


def check_requirements():
    """检查前置条件和依赖"""
    print_header('前置条件检查')
    
    # 检查脚本文件是否存在
    missing_scripts = []
    if RUN_COVERAGE_CHECK and not SCRIPT_CHECK.exists():
        missing_scripts.append(str(SCRIPT_CHECK))
    if RUN_EXTRACTION and not SCRIPT_EXTRACT.exists():
        missing_scripts.append(str(SCRIPT_EXTRACT))
    if RUN_MERGE_PREPROCESS and not SCRIPT_MERGE.exists():
        missing_scripts.append(str(SCRIPT_MERGE))
    
    if missing_scripts:
        print('✗ 缺少必要的脚本文件:')
        for script in missing_scripts:
            print(f'  - {script}')
        return False
    else:
        print('✓ 所有脚本文件都存在')
    
    # 检查工作目录
    if not WORK_DIR.exists():
        print(f'✗ 工作目录不存在: {WORK_DIR}')
        return False
    else:
        print(f'✓ 工作目录存在: {WORK_DIR}')
    
    # 检查输入文件
    input_csv = Path(INPUT_CSV)
    if RUN_EXTRACTION and not input_csv.exists():
        print(f'⚠ 输入文件不存在: {input_csv}')
        print('  请先准备输入数据')
        return False
    elif RUN_EXTRACTION:
        print(f'✓ 输入文件存在: {input_csv}')
    
    return True


def run_script_direct(module_name, description):
    """
    直接导入并运行模块的主函数（避免subprocess编码问题）
    入参:
    - module_name (str): 模块名称（如 'extract_sentinel_band_values'）
    - description (str): 脚本描述
    出参:
    - success (bool): 是否成功运行
    """
    print_header(f'执行: {description}')
    
    try:
        # 动态导入模块
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            module_name, 
            CODE_DIR / f'{module_name}.py'
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 调用主函数
        if hasattr(module, 'main'):
            module.main()
            print('\n✓ 脚本执行成功')
            return True
        else:
            print('✗ 模块没有main函数')
            return False
            
    except Exception as e:
        print(f'✗ 运行脚本时出错: {str(e)}')
        import traceback
        traceback.print_exc()
        return False


def run_script(script_path, description):
    """
    使用subprocess运行Python脚本（保留作为备选）
    入参:
    - script_path (Path): 脚本文件路径
    - description (str): 脚本描述
    出参:
    - success (bool): 是否成功运行
    """
    print_header(f'执行: {description}')
    
    if not script_path.exists():
        print(f'✗ 脚本文件不存在: {script_path}')
        return False
    
    print(f'脚本路径: {script_path}')
    print(f'工作目录: {WORK_DIR}')
    
    try:
        # 运行脚本
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(WORK_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding=SYSTEM_ENCODING,
            errors='replace'  # 使用 'replace' 处理编码错误
        )
        
        # 实时打印输出
        for line in process.stdout:
            print(line, end='')
        
        # 等待完成
        return_code = process.wait()
        
        # 检查结果
        if return_code == 0:
            print('\n✓ 脚本执行成功')
            return True
        else:
            print(f'\n✗ 脚本执行失败，返回码: {return_code}')
            stderr = process.stderr.read()
            if stderr:
                print(f'错误信息:\n{stderr}')
            return False
            
    except Exception as e:
        print(f'✗ 运行脚本时出错: {str(e)}')
        return False


def check_extraction_progress():
    """检查波段值提取进度"""
    print_header('检查提取进度')
    
    # 查找临时文件
    temp_files = glob.glob(str(WORK_DIR / TEMP_PATTERN))
    
    if len(temp_files) == 0:
        print('× 未找到批次临时文件')
        print('  说明: 提取步骤尚未开始或已完成并清理')
        return False
    
    print(f'✓ 找到 {len(temp_files)} 个批次文件')
    
    # 统计每个批次的大小
    total_size = 0
    for temp_file in sorted(temp_files):
        file_size = os.path.getsize(temp_file)
        total_size += file_size
        print(f'  - {os.path.basename(temp_file)}: {file_size / 1024:.2f} KB')
    
    print(f'\n总计: {len(temp_files)} 个批次文件, {total_size / 1024:.2f} KB')
    
    return True


def check_final_results():
    """检查最终结果文件"""
    print_header('检查最终结果')
    
    files_to_check = {
        OUTPUT_EXTRACT: '波段值提取结果',
        OUTPUT_MERGED: '合并完整数据',
        OUTPUT_PROCESSED: '预处理后数据',
        OUTPUT_REPORT: '数据分析报告'
    }
    
    all_exist = True
    for file_path, description in files_to_check.items():
        if file_path.exists():
            file_size = file_path.stat().st_size
            print(f'✓ {description}: {file_path.name} ({file_size / 1024:.2f} KB)')
        else:
            print(f'✗ {description}: 文件不存在')
            all_exist = False
    
    return all_exist


def main():
    """主函数"""
    print_header('湖库水深数据提取与预处理完整流程')
    
    start_time = time.time()
    
    # 1. 前置条件检查
    if not check_requirements():
        print('\n✗ 前置条件检查失败，请解决问题后重试')
        sys.exit(1)
    
    # 2. 可选：检查影像覆盖情况
    if RUN_COVERAGE_CHECK:
        success = run_script_direct('check_image_coverage_only', '影像覆盖检查')
        if not success:
            print('\n⚠ 覆盖率检查失败，但继续后续流程')
            response = input('是否继续执行提取步骤? (y/n): ')
            if response.lower() != 'y':
                sys.exit(1)
    
    # 3. 提取Sentinel-2波段值
    if RUN_EXTRACTION:
        # 检查是否已有部分结果
        temp_files = glob.glob(str(WORK_DIR / TEMP_PATTERN))
        if temp_files:
            print(f'\n⚠ 检测到已有 {len(temp_files)} 个批次文件')
            response = input('是否重新提取? (y=重新提取, n=跳过提取, 直接进入合并): ').lower()
            if response == 'y':
                print('开始重新提取...')
                success = run_script_direct('extract_sentinel_band_values', 'Sentinel-2波段值提取')
                if not success:
                    print('✗ 提取失败')
                    sys.exit(1)
            else:
                print('跳过提取步骤，使用现有批次文件')
        else:
            success = run_script_direct('extract_sentinel_band_values', 'Sentinel-2波段值提取')
            if not success:
                print('✗ 提取失败')
                sys.exit(1)
        
        # 检查提取进度
        check_extraction_progress()
    
    # 4. 合并和预处理
    if RUN_MERGE_PREPROCESS:
        success = run_script_direct('merge_and_preprocess', '数据合并与预处理')
        if not success:
            print('✗ 合并和预处理失败')
            sys.exit(1)
    
    # 5. 检查最终结果
    results_exist = check_final_results()
    
    # 6. 总结
    elapsed_time = time.time() - start_time
    
    print_header('流程执行完成')
    
    if results_exist:
        print('✓ 所有输出文件已生成')
    else:
        print('⚠ 部分输出文件缺失，请检查')
    
    print(f'\n总耗时: {elapsed_time/60:.2f} 分钟')
    
    # 显示关键文件
    print('\n生成的文件:')
    
    visualization_files = glob.glob(str(WORK_DIR / '*.png'))
    for viz_file in sorted(visualization_files):
        file_name = os.path.basename(viz_file)
        if '相关性热图' in file_name or '可视化' in file_name:
            file_size = os.path.getsize(viz_file)
            print(f'  - {file_name} ({file_size / 1024:.2f} KB)')
    
    print('\n建议下一步:')
    print('  1. 检查 数据分析报告.txt 了解数据质量')
    print('  2. 查看可视化图表评估数据分布')
    print('  3. 使用 湖库波段值_预处理后.csv 进行机器学习建模')
    print('=' * 80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n\n⚠ 用户中断')
        sys.exit(1)
    except Exception as e:
        print(f'\n✗ 发生错误: {str(e)}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

