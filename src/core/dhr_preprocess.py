import os
import pandas as pd


data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
query_name = "class_all"
seq_name = "class_all"
method_benchmark = "dhr"

DECOY_METHODS = ["extended_shuf"]

for decoy_method in DECOY_METHODS:
    print(f"Processing {decoy_method}...")
    df_d = pd.read_csv(f"{data_dir}/result_{method_benchmark}_{query_name}_{decoy_method}_{seq_name}.txt", sep="\t")
    df_t = pd.read_csv(f"{data_dir}/result_{method_benchmark}_{query_name}_{seq_name}.txt", sep="\t")

    df_d['score'] = 150-df_d['score']
    df_t['score'] = 150-df_t['score']

    df_d.to_csv(f"{data_dir}/result_{method_benchmark}_postprocess_{query_name}_{decoy_method}_{seq_name}.txt", sep="\t", index=False)
    df_t.to_csv(f"{data_dir}/result_{method_benchmark}_postprocess_{query_name}_{seq_name}.txt", sep="\t", index=False)

    print(df_d.head())
    print(df_t.head())

    print(f'Saved to: {data_dir}/result_{method_benchmark}_postprocess_{query_name}_{decoy_method}_{seq_name}.txt')
    print(f'Saved to: {data_dir}/result_{method_benchmark}_postprocess_{query_name}_{seq_name}.txt')