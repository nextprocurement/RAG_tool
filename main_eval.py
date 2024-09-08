

import argparse
import pathlib
import time
import numpy as np
import pandas as pd
from scipy import sparse
import yaml
from src.evaluation.labeller import TopicLabeller
from src.utils.tm_utils import create_model
from src.evaluation.tm_matcher import TMMatcher
from src.utils.utils import init_logger
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pandas as pd


def load_config(config_path):
    """
    Load configuration from YAML.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    
    # ***********************************************************************
    # Parse arguments
    # ***********************************************************************
    parser = argparse.ArgumentParser(description="HERMES evaluation")
    parser.add_argument(
        "--model_1_path", # Optimized
        help="Path to the first model",
        required=False,
        default="/export/usuarios_ml4ds/lbartolome/Repos/repos_con_carlos/RAG_tool/data/out/optimized/4.training/datos_modelo_es_Mallet_df_merged_14_topics_45_ENTREGABLE/MalletLda_20"
    )
    parser.add_argument(
        "--model_2_path", # Non-optimized
        help="Path to the second model",
        required=False,
        default="/export/usuarios_ml4ds/lbartolome/Repos/repos_con_carlos/RAG_tool/data/out/non_optimized/4.training/datos_modelo_es_Mallet_df_merged_14_topics_45_ENTREGABLE/MalletLda_20"
    )
    parser.add_argument(
        "--path_raw_corpus",
        help="Path to the raw corpus",
        required=False,
        default="/export/usuarios_ml4ds/lbartolome/Repos/repos_con_carlos/RAG_tool/data/out/optimized/2.preprocessing/datos_modelo_es_Mallet_df_merged_14_topics_45_ENTREGABLE_embeddings.parquet"
    )
    parser.add_argument(
        "--id_fld",
        help="Field name for the id",
        required=False,
        default="id_tm"
    )    
    parser.add_argument(
        "--model_types",
        help="Types of the models",
        required=False,
        default="MalletLda,MalletLda"
    )
        
        
    # ***********************************************************************
    # Setup
    # ***********************************************************************
    config_path = pathlib.Path("config/settings.yaml")
    config = load_config(config_path)
    logger = init_logger(config['logger'])
    args = parser.parse_args()
    
    labeller = TopicLabeller(logger=logger)
    
    # ***********************************************************************
    # Load raw corpus
    # ***********************************************************************
    df_raw_corpus = pd.read_parquet(args.path_raw_corpus)
    
    # ***********************************************************************
    # Get models topics, betas and thetas
    # ***********************************************************************
    models = [pathlib.Path(args.model_1_path), pathlib.Path(args.model_2_path)]
    model_types = args.model_types.split(",")
    
    models_betas = []
    models_thetas = []
    models_topics = []
    models_vocabs = []
    for m in range(len(models)):
        params_inference = {
            'load_data_path': args.path_raw_corpus,
            'model_path': models[m],
            'load_model': True,
        }

        model = create_model(model_types[m], **params_inference)
        topics = model.print_topics()
        topics = [topics[topic] for topic in topics]
        
        models_topics.append(topics)
        models_betas.append(model.get_betas())
        models_thetas.append(model.get_thetas())
        models_vocabs.append(model.vocab)
        
    tm_matcher = TMMatcher()
    matches = tm_matcher.iterative_matching(models_topics, 3)

    print("** Matches found:")
    for match in matches:
        print("-- -- Optimized model topic:", match[0][1], models_topics[0][match[0][1]])
        print("-- -- Non-optimized model topic:", match[1][1], models_topics[1][match[1][1]])
        print("-+-"*20)
    
    # ***********************************************************************
    # Get most representative documents for each topic based on S3
    # ***********************************************************************
    print("-- -- Calculating S3...")
    
    for m in range(len(models)):
        betas = models_betas[m]
        thetas = models_thetas[m]
        
        ##########
        # CORPUS #
        ##########
        # TODO: Encontrar el fichero que es parquet
        
        df_corpus = pd.read_parquet("/export/usuarios_ml4ds/lbartolome/Repos/repos_con_carlos/RAG_tool/data/out/optimized/4.training/datos_modelo_es_Mallet_df_merged_14_topics_45_ENTREGABLE/MalletLda_20/datos_modelo_es_Mallet_df_merged_14_topics_45_ENTREGABLE_embeddings_preproc.parquet")
        
        ##########
        # VOCAB #
        ##########
        vocab_w2id = {}
        for i, line in enumerate(models_vocabs[m]):
            wd = line.strip()
            vocab_w2id[wd] = i
        
        start = time.time()
        S3 = np.zeros((len(thetas), len(betas)))

        # For each document
        for doc in range(len(thetas)):
            # For each topic
            for topic in range(thetas.shape[1]):

                # ids of the words of document doc in the vocabulary
                wd_ids = []
                for word in corpus[doc]:
                    try:
                        wd_ids.append(vocab_w2id[word])
                    except Exception as e:
                        #print(f"Word {word} not found in vocabulary") 
                        continue

                # sum of the weights that topic assings to each word in the document
                try:
                    S3[doc, topic] = np.sum(betas[topic, wd_ids])
                except Exception as e:
                    #print(f"Error: {e}")
                    import pdb; pdb.set_trace()

        print(f"S3 shape: {S3.shape}")

        # S3_sparse = sparse.csr_matrix(S3)
        print(f"Time elapsed: {time.time() - start}")

        # Find the most representative document for each topic
        top_docs_per_topic = []

        for s3_ in S3.T:  
            sorted_docs_indices = np.argsort(s3_)[::-1]  ## Sort the documents based on their proportion for the current topic in descending order
            top = sorted_docs_indices[:3]
            top_docs_per_topic.append(top)

        # get raw_text for each top doc
        top_docs_per_topic_text = []
        for topic_docs in top_docs_per_topic:
            docs = []
            for doc in topic_docs:
                try:
                    docs.append(df_corpus.iloc[doc].raw_text)
                except Exception as e:
                    docs.append("Document not found")
                    import pdb; pdb.set_trace()
            #docs = [df_corpus[df_corpus.id_tm == doc].iloc[0].raw_text for doc in topic_docs]
            top_docs_per_topic_text.append(docs)

        top_docs_0 = [docs[0] for docs in top_docs_per_topic_text]
        top_docs_1 = [docs[1] for docs in top_docs_per_topic_text]
        top_docs_2 = [docs[2] for docs in top_docs_per_topic_text]
        
        # Find the most representative document for each topic based on thetas
        top_docs_per_topic = []

        for theta in thetas.T:  
            sorted_docs_indices = np.argsort(theta)[::-1]  ## Sort the documents based on their proportion for the current topic in descending order
            top = sorted_docs_indices[:3]
            top_docs_per_topic.append(top)

        # get raw_text for each top doc
        top_docs_per_topic_text = []
        for topic_docs in top_docs_per_topic:
            for doc in topic_docs:
                try:
                    docs.append(df_corpus.iloc[doc].raw_text)
                except Exception as e:
                    docs.append("Document not found")
                    import pdb; pdb.set_trace()
            #docs = [df_corpus[df_corpus.id_tm == doc].iloc[0].raw_text for doc in topic_docs]
            top_docs_per_topic_text.append(docs)

        top_docs_0_thetas = [docs[0] for docs in top_docs_per_topic_text]
        top_docs_1_thetas = [docs[1] for docs in top_docs_per_topic_text]
        top_docs_2_thetas = [docs[2] for docs in top_docs_per_topic_text]
        
        # Get topic labels
        topic_labels = labeller.label_topics(models_topics[m], path_save=f"{models[m]}/topic_labels.txt")

        df = pd.DataFrame(
            {
                "Topic ID": range(len(models_topics[m])),
                "Topic Label": topic_labels,
                "Chemical description": models_topics[m],
                "S3 most representative document #1": top_docs_0,
                "S3 most representative document #2": top_docs_1,
                "S3 most representative document #3": top_docs_2,
                "thetas most representative document #1": top_docs_0_thetas,
                "thetas most representative document #2": top_docs_1_thetas,
                "thetas most representative document #3": top_docs_2_thetas,
            }
        )

        path_save = f"{models[m]}/most_representative_docs_per_topic.xlsx"
        df.to_excel(path_save)
    
if __name__ == "__main__":
   main() 