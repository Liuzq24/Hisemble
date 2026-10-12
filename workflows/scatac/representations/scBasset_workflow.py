import os
import subprocess
import time
import psutil
from pathlib import Path

# Dataset and training parameters
# Each configuration dictionary defines one independent training task.
# Set dataset-specific parameters in the configuration list.
TRAINING_JOBS = [
    {
        'dataname': 'FCA_fetal_33184180_eye',
        'input_folder': '__HISEMBLE_DATA_ROOT__/scBasset/FCA_fetal_33184180_eye_processed',
        'bottleneck': 32,
        'batch_size': 128,
        'lr': 0.01,
        'epochs': 1000, # Use a small value for this example.
    },
    {
        'dataname': 'FCA_fetal_33184180_heart',
        'input_folder': '__HISEMBLE_DATA_ROOT__/scBasset/FCA_fetal_33184180_heart_processed',
        'bottleneck': 32, # Example: use a different bottleneck size for this dataset.
        'batch_size': 128,
        'lr': 0.01,
        'epochs': 1000,
    },

]

# Define the root output directory.
BASE_OUTPUT_DIR = Path('__HISEMBLE_DATA_ROOT__/scBasset_output')

# 2. Execute each training task.
for job in TRAINING_JOBS:
    dataname = job['dataname']
    print(f"\n{'='*20} Processing dataset: {dataname} {'='*20}")

    # 3. Create an output directory for the current dataset.
    output_path = BASE_OUTPUT_DIR / dataname
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Outputs will be saved to: {output_path}")
    log_file_path = output_path / f'training_log_{dataname}.txt'

    # 4. Prepare command-line arguments for scbasset_train.py.
    # Pass arguments as a list instead of interpolating a shell command.
    command = [
        'python',              
        'scbasset_train.py',
        '--input_folder', str(job['input_folder']),
        '--out_path', str(output_path),
        '--bottleneck', str(job['bottleneck']),
        '--batch_size', str(job['batch_size']),
        '--lr', str(job['lr']),
        '--epochs', str(job['epochs']),
        # Add --trained_model or --test here when needed.
        # '--test'  # Uncomment to run in test mode.
    ]

    # 5. Monitor performance.
    # Get the current process.
    process = psutil.Process(os.getpid())
    # Record initial memory usage.
    mem_before_mb = process.memory_info().rss / (1024 * 1024)
    # Record the start time.
    start_time = time.perf_counter()

    print(f"\nExecuting command: {' '.join(command)}\n")

    # 6. Launch the training script with subprocess.
    # subprocess.run waits for the child process to finish.
    with open(log_file_path, 'w') as log_file:
        try:
            # Use Popen to capture output while training runs.
            with subprocess.Popen(command, 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.STDOUT, # Merge standard error into standard output.
                                  text=True, 
                                  bufsize=1, 
                                  encoding='utf-8') as proc:
                
                # Read child-process output as it is produced.
                for line in proc.stdout:
                    # 1. Display output in the terminal for monitoring.
                    print(line, end='')
                    # 2. Write the same output to the log file.
                    log_file.write(line)
            
            # Check whether the child process succeeded.
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, proc.args)

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            error_message = f"\nError: training failed for dataset {dataname}.\n{str(e)}\n"
            print(error_message)
            log_file.write(error_message)
            # Continue with the next task if the current task fails.
            continue

    end_time = time.perf_counter()
    mem_after_mb = process.memory_info().rss / (1024 * 1024)

    elapsed_time_s = end_time - start_time
    main_script_mem_mb = mem_after_mb - mem_before_mb
    # 7. Record and print performance metrics.
    performance_report = f"""
--------------------------------------------------
Performance report for {dataname}
--------------------------------------------------
Total wall time: {elapsed_time_s:.2f} s ({elapsed_time_s/60:.2f} min)
Controller script memory increase: {main_script_mem_mb:.2f} MB
Detailed training memory measurements appear in the preceding log entries.
--------------------------------------------------
"""
    
    # Print the performance report.
    print(performance_report)
    
    # Append the performance report to the dataset log file.
    with open(log_file_path, 'a') as log_file:
        log_file.write(performance_report)

print(f"\n{'='*20} All tasks completed {'='*20}")