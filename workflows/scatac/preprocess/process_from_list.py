import os
import scanpy as sc
import episcanpy as epi
import numpy as np

def process_h5ad_file(input_path, output_dir, filter_ratio):
    """Binarize and filter features in one h5ad file, then save the result."""
    try:
        filename = os.path.basename(input_path)
        print(f"--- Processing: {filename} ---")

        adata = sc.read(input_path)
        print(f"Original shape (cells x features): {adata.shape}")
        if adata.shape[0] == 0 or adata.shape[1] == 0:
            print("Warning: empty input or invalid dimensions; skipping this file.")
            return False

        print("Step 1/2: binarizing the data...")
        epi.pp.binarize(adata)

        min_cells = np.ceil(filter_ratio * adata.shape[0])
        print(f"Step 2/2: retaining features detected in at least {int(min_cells)} cells...")
        epi.pp.filter_features(adata, min_cells=min_cells)
        
        print(f"Processed shape (cells x features): {adata.shape}")

        base_name, ext = os.path.splitext(filename)
        output_filename = f"{base_name}_processed{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Saving to: {output_path}")
        adata.write(output_path, compression='gzip')

        print(f"--- {filename} processed successfully ---\n")
        return True

    except Exception as e:
        print(f"Error while processing {input_path}: {e}\n")
        return False

# ===================================================================
# Main execution block.
# ===================================================================
if __name__ == "__main__":
    file_paths_to_process = [
        '__HISEMBLE_RAW_ATAC_ROOT__/sciCAR_mm10_kdiney_ATAC.h5ad'
    ]

    output_directory = '__HISEMBLE_DATA_ROOT__/0.01_h5ad/'
    filter_ratio = 0.01

    
    print("--- Starting batch processing ---")
    print(f"Processing {len(file_paths_to_process)} files from the input list.")
    print(f"Output directory: {output_directory}")
    print(f"Feature-filtering fraction: {filter_ratio * 100}%\n")

    os.makedirs(output_directory, exist_ok=True)
    
    failed_files = []
    
    if not file_paths_to_process or 'path/to/your/data' in file_paths_to_process[0]:
        print("Error: populate file_paths_to_process with your input file paths before running the script.")
    else:
        for file_path in file_paths_to_process:
            # Check that the input file exists before processing.
            if not os.path.exists(file_path):
                print(f"File not found; skipping: {file_path}\n")
                failed_files.append(f"{file_path} (file not found)")
                continue

            success = process_h5ad_file(file_path, output_directory, filter_ratio)
            
            if not success:
                failed_files.append(file_path)

    # Final report.
    print("="*50)
    print("Batch processing completed.")
    total_files = len(file_paths_to_process)
    success_count = total_files - len(failed_files)
    print(f"Summary: {total_files} files; {success_count} succeeded; {len(failed_files)} failed.")
    
    if failed_files:
        print("\nThe following files failed or were not found:")
        for f in failed_files:
            print(f"- {f}")
    else:
        print("\nAll files were processed successfully.")
    print("="*50)