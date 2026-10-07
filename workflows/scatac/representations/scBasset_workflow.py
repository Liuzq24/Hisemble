import os
import subprocess
import time
import psutil
from pathlib import Path

# Dataset and training parameters
# 这是一个配置列表，每个元素都是一个字典，代表一次独立的训练任务。
# 这样做非常灵活，您可以为每个数据集设置不同的参数。
TRAINING_JOBS = [
    {
        'dataname': 'FCA_fetal_33184180_eye',
        'input_folder': '__HISEMBLE_DATA_ROOT__/scBasset/FCA_fetal_33184180_eye_processed',
        'bottleneck': 32,
        'batch_size': 128,
        'lr': 0.01,
        'epochs': 1000, # 为了演示，设置一个较小的值
    },
    {
        'dataname': 'FCA_fetal_33184180_heart',
        'input_folder': '__HISEMBLE_DATA_ROOT__/scBasset/FCA_fetal_33184180_heart_processed',
        'bottleneck': 32, # 示例：为这个数据集使用不同的瓶颈层大小
        'batch_size': 128,
        'lr': 0.01,
        'epochs': 1000,
    },

]

# 定义所有输出的根目录
BASE_OUTPUT_DIR = Path('__HISEMBLE_DATA_ROOT__/scBasset_output')

# --- 2. 主循环，执行每个训练任务 ---
for job in TRAINING_JOBS:
    dataname = job['dataname']
    print(f"\n{'='*20} 开始处理数据集: {dataname} {'='*20}")

    # --- 3. 为当前数据集创建专属的输出目录 ---
    output_path = BASE_OUTPUT_DIR / dataname
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"输出将保存到: {output_path}")
    log_file_path = output_path / f'training_log_{dataname}.txt'

    # --- 4. 准备调用 scbasset_train.py 的命令行参数 ---
    # 我们将以列表的形式构建命令，这比拼接字符串更安全
    command = [
        'python',              
        'scbasset_train.py',
        '--input_folder', str(job['input_folder']),
        '--out_path', str(output_path),
        '--bottleneck', str(job['bottleneck']),
        '--batch_size', str(job['batch_size']),
        '--lr', str(job['lr']),
        '--epochs', str(job['epochs']),
        # 如果需要，可以在这里添加 --trained_model 或 --test 等其他参数
        # '--test' # 例如，如果想在测试模式下运行，取消这行注释
    ]

    # --- 5. 性能监控 ---
    # 获取当前进程对象
    process = psutil.Process(os.getpid())
    # 记录开始前的内存
    mem_before_mb = process.memory_info().rss / (1024 * 1024)
    # 记录开始时间
    start_time = time.perf_counter()

    print(f"\n正在执行命令: {' '.join(command)}\n")

    # --- 6. 使用 subprocess 调用训练脚本 ---
    # subprocess.run 会等待子进程（即训练脚本）执行完毕
    with open(log_file_path, 'w') as log_file:
        try:
            # 使用 Popen 来执行子进程，这样我们可以实时捕获输出
            with subprocess.Popen(command, 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.STDOUT, # 将标准错误也合并到标准输出
                                  text=True, 
                                  bufsize=1, 
                                  encoding='utf-8') as proc:
                
                # 实时读取子进程的输出流
                for line in proc.stdout:
                    # 1. 实时打印到当前终端，方便您监控
                    print(line, end='')
                    # 2. 同时将这行内容写入日志文件
                    log_file.write(line)
            
            # 检查子进程是否成功执行
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, proc.args)

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            error_message = f"\n!!!!!! 错误：数据集 {dataname} 的训练失败 !!!!!!\n{str(e)}\n"
            print(error_message)
            log_file.write(error_message)
            # 即使当前任务失败，也继续处理下一个任务
            continue

    end_time = time.perf_counter()
    mem_after_mb = process.memory_info().rss / (1024 * 1024)

    elapsed_time_s = end_time - start_time
    main_script_mem_mb = mem_after_mb - mem_before_mb
    # --- 7. 记录并打印性能指标 ---
    performance_report = f"""
--------------------------------------------------
性能报告 (Performance Report for {dataname})
--------------------------------------------------
总耗时 (Total Wall Time): {elapsed_time_s:.2f} 秒 ({elapsed_time_s/60:.2f} 分钟)
主控脚本内存增量 (Controller Script Memory Increase): {main_script_mem_mb:.2f} MB
(注意: 训练过程中的详细内存使用情况请查看日志文件上方的 'Memory usage' 打印)
--------------------------------------------------
"""
    
    # 打印性能报告到终端
    print(performance_report)
    
    # 将性能报告追加到该数据集的日志文件末尾
    with open(log_file_path, 'a') as log_file:
        log_file.write(performance_report)

print(f"\n{'='*20} 所有任务处理完毕 {'='*20}")