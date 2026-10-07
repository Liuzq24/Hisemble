import os
import re
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 路径与参数配置
# ==========================================
GTF_PATH = '__HISEMBLE_ANNOTATION_ROOT__/gencode.vM25.annotation.gtf'
# 替换为你实际的 BED 文件路径
BED_PATH = 'cluster_0_GREAT.bed' 
OUT_DIR = './GO'
OUT_CSV = os.path.join(OUT_DIR, 'mapped_peaks_to_genes.csv') # 输出的 CSV 路径

os.makedirs(OUT_DIR, exist_ok=True)
PROMOTER_LEN = 2000 # TSS 上游 2kb

# ==========================================
# 2. 解析 GTF 文件
# ==========================================
def load_gene_regions(gtf_path, promoter_len=2000):
    print(f"正在解析 GTF 文件 (提取 TSS-{promoter_len}bp 到 TES)...")
    gene_regions = []
    
    with open(gtf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split('\t')
            if len(parts) < 9 or parts[2] != 'gene': continue
                
            chrom = parts[0]
            start = int(parts[3]) - 1 # GTF 是 1-based, 转为 0-based
            end = int(parts[4])
            strand = parts[6]
            
            # 提取基因名
            match = re.search(r'gene_name "([^"]+)"', parts[8])
            if not match: continue
            gene_name = match.group(1)
            
            # 根据正负链计算 [TSS-2kb, TES]
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
    print(f"GTF 解析完成，共提取 {len(df_genes)} 个基因的靶向区间。")
    return df_genes

# ==========================================
# 3. Peak to Gene 映射
# ==========================================
def map_peaks_to_genes(peaks_df, genes_df):
    print("正在进行 Peak 到 Gene 的物理位置映射...")
    
    mapping_results = []
    
    # 按染色体分组处理，极大提高速度
    for chrom, p_group in peaks_df.groupby('chrom'):
        g_group = genes_df[genes_df['chrom'] == chrom]
        if g_group.empty: continue
            
        for _, p_row in p_group.iterrows():
            p_start, p_end = p_row['start'], p_row['end']
            
            # 判断交集：Peak的起点 <= Gene的终点 且 Peak的终点 >= Gene的起点
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
    print(f"映射完成！共建立 {len(mapped_df)} 对 Peak-Gene 连接。")
    return mapped_df

# ==========================================
# 4. 主程序运行逻辑
# ==========================================
if __name__ == '__main__':
    # 1. 加载 GTF 基因组注释
    df_genes = load_gene_regions(GTF_PATH, PROMOTER_LEN)
    
    # 2. 读取 BED 文件 
    # 使用 \t 分隔，不包含表头，直接手动命名列名为 'chrom', 'start', 'end', 'peak'
    print(f"正在读取 BED 文件: {BED_PATH}")
    df_peaks = pd.read_csv(BED_PATH, sep='\t', header=None, names=['chrom', 'start', 'end', 'peak'])
    
    # 3. 进行映射
    mapped_df = map_peaks_to_genes(df_peaks, df_genes)
    
    # 4. 保存为 CSV
    if not mapped_df.empty:
        mapped_df.to_csv(OUT_CSV, index=False)
        print(f"结果已成功保存至: {OUT_CSV}")
    else:
        print("警告：未找到任何匹配的 Peak-Gene 对，没有生成 CSV 文件。")