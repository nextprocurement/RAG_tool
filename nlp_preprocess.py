import argparse
import pathlib
import subprocess
from tqdm import tqdm

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=False, help="Path to the input file", default="/export/usuarios_ml4ds/cggamella/NP-Search-Tool/sample_data/models/Mallet/es_Mallet_df_merged_20_topics_45_ENTREGABLE/datos_modelo.parquet")
    parser.add_argument("--output", type=str, required=False, help="Path to the output file", default="/export/usuarios_ml4ds/lbartolome/Repos/repos_con_carlos/RAG_tool/data/preprocessed/cpv_45_preproc.parquet")
    
    args = parser.parse_args()
        

    preprocessing_script = "/export/usuarios_ml4ds/lbartolome/Repos/repos_con_carlos/RAG_tool/src/preprocessing/pipe/nlpipe.py"
    source_path = args.input
    source_type = "parquet"
    source = "cpv45"
    destination_path = args.output
    spacy_model = "es_core_news_lg"
    lang = "es"

    # Construct the command
    cmd = [
        "python", preprocessing_script,
        "--source_path", source_path,
        "--source_type", source_type,
        "--source", source,
        "--destination_path", destination_path,
        "--lang", lang,
        "--spacy_model", spacy_model,
        "--do_embeddings"
    ]

    try:
        print(f'-- -- Running preprocessing command {" ".join(cmd)}')
        subprocess.check_output(cmd)
    except subprocess.CalledProcessError as e:
        print('-- -- Preprocessing failed. Revise command')
        print(e.output)
    print("-- -- Preprocessing done")
