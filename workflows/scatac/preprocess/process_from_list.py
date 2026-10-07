import os
import scanpy as sc
import episcanpy as epi
import numpy as np

def process_h5ad_file(input_path, output_dir, filter_ratio):
    """
    处理单个h5ad文件：二值化、特征筛选，并保存结果。
    """
    try:
        filename = os.path.basename(input_path)
        print(f"--- 开始处理: {filename} ---")

        adata = sc.read(input_path)
        print(f"原始维度 (细胞 x 特征): {adata.shape}")
        if adata.shape[0] == 0 or adata.shape[1] == 0:
            print("警告: 文件为空或维度不正确，跳过处理。")
            return False

        print("步骤 1/2: 正在进行数据二值化...")
        epi.pp.binarize(adata)

        min_cells = np.ceil(filter_ratio * adata.shape[0])
        print(f"步骤 2/2: 筛选特征，保留在 >= {int(min_cells)} 个细胞中存在的特征...")
        epi.pp.filter_features(adata, min_cells=min_cells)
        
        print(f"处理后维度 (细胞 x 特征): {adata.shape}")

        base_name, ext = os.path.splitext(filename)
        output_filename = f"{base_name}_processed{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"正在保存到: {output_path}")
        adata.write(output_path, compression='gzip')

        print(f"--- {filename} 处理成功 ---\n")
        return True

    except Exception as e:
        print(f"!!! 处理文件 {input_path} 时发生错误: {e} !!!\n")
        return False

# ===================================================================
#                      主程序执行部分
# ===================================================================
if __name__ == "__main__":
    file_paths_to_process = [
        '__HISEMBLE_RAW_ATAC_ROOT__/sciCAR_mm10_kdiney_ATAC.h5ad'
    ]

    output_directory = '__HISEMBLE_DATA_ROOT__/0.01_h5ad/'
    filter_ratio = 0.01

    
    print("--- 批量处理脚本启动 ---")
    print(f"将从列表中处理 {len(file_paths_to_process)} 个文件。")
    print(f"输出目录: {output_directory}")
    print(f"特征筛选比例: {filter_ratio * 100}%\n")

    os.makedirs(output_directory, exist_ok=True)
    
    failed_files = []
    
    if not file_paths_to_process or 'path/to/your/data' in file_paths_to_process[0]:
        print("错误：请先编辑脚本，在 `file_paths_to_process` 列表中填入您的实际文件路径。")
    else:
        for file_path in file_paths_to_process:
            # 检查文件是否存在，避免不必要的错误
            if not os.path.exists(file_path):
                print(f"!!! 文件未找到，跳过: {file_path} !!!\n")
                failed_files.append(f"{file_path} (文件不存在)")
                continue

            success = process_h5ad_file(file_path, output_directory, filter_ratio)
            
            if not success:
                failed_files.append(file_path)

    # --- 最终报告 ---
    print("="*50)
    print("批量处理完成！")
    total_files = len(file_paths_to_process)
    success_count = total_files - len(failed_files)
    print(f"总结: 共处理 {total_files} 个文件，{success_count} 个成功，{len(failed_files)} 个失败。")
    
    if failed_files:
        print("\n以下文件处理失败或未找到，请检查：")
        for f in failed_files:
            print(f"- {f}")
    else:
        print("\n所有文件均已成功处理！")
    print("="*50)