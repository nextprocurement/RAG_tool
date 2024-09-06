import argparse
import pathlib
import subprocess
from tqdm import tqdm

'''
python3 nlp_preprocess.py --input /export/usuarios_ml4ds/cggamella/RAG_tool/files/anotacion_manual/fam/df_esp_first100.xlsx
--output /export/usuarios_ml4ds/cggamella/RAG_tool/data/preprocessed/manual_fam_df_esp_first100.xlsx
--path_add_acr /export/usuarios_ml4ds/cggamella/RAG_tool/topicmodelling/data/acronyms/df_esp_first100_both_equivalences.json
'''

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=False, help="Path to the input file", default="/export/usuarios_ml4ds/cggamella/NP-Search-Tool/sample_data/models/Mallet/es_Mallet_df_merged_20_topics_45_ENTREGABLE/datos_modelo.parquet")
    parser.add_argument("--output", type=str, required=False, help="Path to the output file", default="/export/usuarios_ml4ds/cggamella/RAG_tool/data/preprocessed/cpv_45_preproc.parquet")
    parser.add_argument("--path_add_acr",  type=str, required=False, help="Path to the acr file", default="/export/usuarios_ml4ds/cggamella/RAG_tool/topicmodelling/data/acronyms/df_esp_first100_both_equivalences.json")
    args = parser.parse_args()
        

    preprocessing_script = "/export/usuarios_ml4ds/lbartolome/Repos/repos_con_carlos/RAG_tool/src/preprocessing/pipe/nlpipe.py"
    source_path = args.input
    source_type = "parquet"
    source = "cpv45"
    destination_path = args.output
    spacy_model = "es_core_news_lg"
    lang = "es"

    # Construct the command
    
    if args.path_add_acr:
        cmd = [
            "python", preprocessing_script,
            "--source_path", source_path,
            "--source_type", source_type,
            "--source", source,
            "--destination_path", destination_path,
            "--lang", lang,
            "--spacy_model", spacy_model,
            "--path_add_acr", args.path_add_acr,
            "--do_embeddings"
        ]
        
    else:
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
