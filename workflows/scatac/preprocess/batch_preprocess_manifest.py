#!/usr/bin/env python
import os
import csv
import subprocess
import argparse
from pathlib import Path

def make_manifest_parser():
    """Parses command line arguments for the manifest-based batch processing."""
    parser = argparse.ArgumentParser(
        description="Batch run scbasset_preprocess.py using a manifest file to link data to specific genomes."
    )
    parser.add_argument(
        '--manifest_file', 
        type=str, 
        required=True,
        help="Path to the manifest file (TSV format: /path/to/data.h5ad<TAB>/path/to/genome.fa)."
    )
    parser.add_argument(
        '--output_base_dir', 
        type=str, 
        required=True,
        help="Base directory where output folders will be created. A subfolder will be created for each input file."
    )
    parser.add_argument(
        '--script_path', 
        type=str, 
        default='./scbasset_preprocess.py',
        help="Path to the scbasset_preprocess.py script."
    )
    return parser

def main():
    """Main function to read the manifest and launch preprocessing jobs."""
    parser = make_manifest_parser()
    args = parser.parse_args()

    # --- 1. Validate paths and arguments ---
    if not os.path.isfile(args.manifest_file):
        print(f"Error: Manifest file not found at {args.manifest_file}")
        return
    if not os.path.isfile(args.script_path):
        print(f"Error: Preprocessing script not found at {args.script_path}")
        return

    # Create the base output directory if it doesn't exist
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    # --- 2. Read and process the manifest file ---
    print(f"Reading manifest file from: {args.manifest_file}")
    
    with open(args.manifest_file, 'r') as f:
        # Using csv.reader is robust for tab-separated files
        reader = csv.reader(f, delimiter='\t')
        
        for i, row in enumerate(reader):
            # Skip empty lines or comments
            if not row or row[0].strip().startswith('#'):
                continue
            
            if len(row) != 2:
                print(f"Warning: Skipping malformed line {i+1} in manifest: {row}")
                continue

            ad_file_path = row[0].strip()
            fasta_file_path = row[1].strip()

            # --- 3. For each entry, run the preprocessing script ---
            
            # Check if the files listed in the manifest actually exist
            if not os.path.isfile(ad_file_path):
                print(f"Warning: anndata file '{ad_file_path}' on line {i+1} not found. Skipping.")
                continue
            if not os.path.isfile(fasta_file_path):
                print(f"Warning: Fasta file '{fasta_file_path}' on line {i+1} not found. Skipping.")
                continue

            # Create a unique output directory for this sample
            sample_name = Path(ad_file_path).stem
            output_path = os.path.join(args.output_base_dir, sample_name)
            
            print("-" * 60)
            print(f"Processing Entry {i+1}:")
            print(f"  Input AnnData: {ad_file_path}")
            print(f"  Input Fasta:   {fasta_file_path}")
            print(f"  Outputting to: {output_path}")

            # Construct the command to run
            command = [
                'python',
                args.script_path,
                '--ad_file', ad_file_path,
                '--input_fasta', fasta_file_path, # This now comes from the manifest
                '--out_path', output_path
            ]
            
            try:
                # We use capture_output=True to catch stdout/stderr if an error occurs
                result = subprocess.run(command, check=True, text=True, capture_output=True)
                print(f"Successfully processed {sample_name}.")
                # Optional: print stdout for successful runs if needed
                # print(result.stdout) 
            except subprocess.CalledProcessError as e:
                print(f"!!! ERROR processing {sample_name}. !!!")
                print(f"Return code: {e.returncode}")
                print("--- STDOUT from failed script ---")
                print(e.stdout)
                print("--- STDERR from failed script ---")
                print(e.stderr)
                print("-" * 60)

    print("\nBatch processing complete.")

if __name__ == "__main__":
    main()