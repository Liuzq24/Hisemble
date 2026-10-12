import os
import re
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. Configure paths and parameters.
# ==========================================
GTF_PATH = '__HISEMBLE_ANNOTATION_ROOT__/gencode.vM25.annotation.gtf'
# Set the path to the input BED file.
BED_PATH = 'cluster_0_GREAT.bed' 
OUT_DIR = './GO'
OUT_CSV = os.path.join(OUT_DIR, 'mapped_peaks_to_genes.csv') # Output CSV path.

os.makedirs(OUT_DIR, exist_ok=True)
PROMOTER_LEN = 2000 # Extend 2 kb upstream of the TSS.

# ==========================================
# 2. Parse the GTF file.
# ==========================================
def load_gene_regions(gtf_path, promoter_len=2000):
    print(f"Parsing the GTF file (TSS-{promoter_len} bp through TES)...")
    gene_regions = []
    
    with open(gtf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split('\t')
            if len(parts) < 9 or parts[2] != 'gene': continue
                
            chrom = parts[0]
            start = int(parts[3]) - 1 # Convert one-based GTF coordinates to zero-based coordinates.
            end = int(parts[4])
            strand = parts[6]
            
            # Extract gene names.
            match = re.search(r'gene_name "([^"]+)"', parts[8])
            if not match: continue
            gene_name = match.group(1)
            
            # Calculate strand-aware intervals from 2 kb upstream of TSS through TES.
            if strand == '+':
                ext_start = max(0, start - promoter_len)
                ext_end = end
            else:
                ext_start = start
                ext_end = end + promoter_len
                
            gene_regions.append({
                'chrom': chrom,
                'start': ext_start,
                'end': ext_end,
                'gene': gene_name
            })
            
    df_genes = pd.DataFrame(gene_regions)
    print(f"GTF parsing completed; extracted target intervals for {len(df_genes)} genes.")
    return df_genes

# ==========================================
# 3. Map peaks to genes.
# ==========================================
def map_peaks_to_genes(peaks_df, genes_df):
    print("Mapping peaks to gene intervals by genomic overlap...")
    
    mapping_results = []
    
    # Process intervals by chromosome.
    for chrom, p_group in peaks_df.groupby('chrom'):
        g_group = genes_df[genes_df['chrom'] == chrom]
        if g_group.empty: continue
            
        for _, p_row in p_group.iterrows():
            p_start, p_end = p_row['start'], p_row['end']
            
            # Intervals overlap when peak_start <= gene_end and peak_end >= gene_start.
            overlaps = g_group[(g_group['start'] <= p_end) & (g_group['end'] >= p_start)]
            
            for gene in overlaps['gene'].unique():
                mapping_results.append({
                    'chrom': chrom,
                    'peak_start': p_start,
                    'peak_end': p_end,
                    'peak': p_row['peak'],
                    'gene': gene
                })
                
    mapped_df = pd.DataFrame(mapping_results)
    print(f"Mapping completed; identified {len(mapped_df)} peak-gene links.")
    return mapped_df

# ==========================================
# 4. Main execution block.
# ==========================================
if __name__ == '__main__':
    # 1. Load the GTF genome annotation.
    df_genes = load_gene_regions(GTF_PATH, PROMOTER_LEN)
    
    # 2. Read the BED file.
    # Read tab-separated data without a header; use chrom, start, end and peak columns.
    print(f"Reading BED file: {BED_PATH}")
    df_peaks = pd.read_csv(BED_PATH, sep='\t', header=None, names=['chrom', 'start', 'end', 'peak'])
    
    # 3. Map intervals.
    mapped_df = map_peaks_to_genes(df_peaks, df_genes)
    
    # 4. Save results to CSV.
    if not mapped_df.empty:
        mapped_df.to_csv(OUT_CSV, index=False)
        print(f"Results saved to: {OUT_CSV}")
    else:
        print("Warning: no overlapping peak-gene pairs were found; no CSV file was generated.")