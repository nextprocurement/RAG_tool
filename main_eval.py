

import argparse
import os
import pathlib
import sys
import time
import numpy as np
import pandas as pd
import yaml
from src.evaluation.labeller import TopicLabeller
from src.evaluation.retriever import Index
from src.utils.tm_utils import create_model
from src.evaluation.tm_matcher import TMMatcher
from src.utils.utils import init_logger
import pandas as pd
from sentence_transformers import SentenceTransformer
import glob
from sklearn.metrics.pairwise import cosine_similarity

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
        default="/export/usuarios_ml4ds/lbartolome/Repos/repos_con_carlos/RAG_tool/data/out/optimized/4.training/datos_modelo_es_Mallet_df_merged_14_topics_45_ENTREGABLE/MalletLda_14"
    )
    parser.add_argument(
        "--model_2_path", # Non-optimized
        help="Path to the second model",
        required=False,
        default="/export/usuarios_ml4ds/lbartolome/Repos/repos_con_carlos/RAG_tool/data/out/non_optimized/4.training/datos_modelo_es_Mallet_df_merged_14_topics_45_ENTREGABLE/MalletLda_14"
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
    # Get models topics, betas and thetas
    # ***********************************************************************
    models = [pathlib.Path(args.model_1_path), pathlib.Path(args.model_2_path)]
    model_types = args.model_types.split(",")
    
    models_betas = []
    models_thetas = []
    models_topics = []
    models_vocabs = []
    models_labels = []
    models_most_repr = []
    
    for m in range(len(models)):
        params_inference = {
            'load_data_path': "",
            'model_path': models[m],
            'load_model': True,
        }

        model = create_model(model_types[m], **params_inference)
        if m == 0: # Optimized
            tfidf = False
        else: # Non-optimized
            tfidf = False
        topics = model.print_topics(tfidf=tfidf)
        topics = [topics[topic][:15] for topic in topics]
        
        models_topics.append(topics)
        models_betas.append(model.get_betas())
        models_thetas.append(model.get_thetas())
        models_vocabs.append(model.vocab)
    
    import pdb; pdb.set_trace()
    tm_matcher = TMMatcher()
    #matches = tm_matcher.iterative_matching(models_topics, 5)#len(models_topics[0]))
    """
    for match in matches:
        print("-- -- Optimized model topic:", match[0][1], models_topics[0][match[0][1]])
        print("-- -- Non-optimized model topic:", match[1][1], models_topics[1][match[1][1]])
        print("-+-"*20)
    """
    matches = tm_matcher.one_to_one_matching(models_topics[0],models_topics[1],len(models_topics[0]))

    print("** Matches found:")
    for match in matches:
        print("-- -- Optimized model topic:", match[0], models_topics[0][match[0]])
        print("-- -- Non-optimized model topic:", match[1], models_topics[1][match[1]])
        print("-+-"*20)
        
    import pdb; pdb.set_trace()
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
        
        file_pattern = os.path.join(models[m], '*_preproc.parquet')
        try:
            path_tr_corpus = glob.glob(file_pattern)[0]
        except IndexError as e:
            print(f"Error: {e}")
            sys.exit(f"File not found: {file_pattern}")
        
        df_corpus = pd.read_parquet(path_tr_corpus)
        corpus = [el.split() for el in df_corpus["lemmas"].tolist()]
        
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
                        #print(f"Word {word} not found in vocabulary") –
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
        if pathlib.Path(f"{models[m]}/topic_labels.txt").exists():
            with open(f"{models[m]}/topic_labels.txt", "r") as f:
                topic_labels = f.readlines()
        else:
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
            
            models_most_repr.append(df)

            path_save = f"{models[m]}/most_representative_docs_per_topic.xlsx"
            df.to_excel(path_save)
        models_labels.append(topic_labels)
    
    # ***********************************************************************
    # Index
    # ***********************************************************************
    logger.info("-- -- Indexing corpus...")
    time_start = time.time()
    hermes_index = Index(df_corpus.raw_text.tolist(), df_corpus.id_tm.tolist())
    logger.info(f"-- -- Corpus indexed in {time.time()-time_start} minutes")
    
    print("** Matches found:")
    all_info = []
    for match in matches:
        
        match_0 = match[0]
        match_0_tpc_desc = models_topics[0][match_0]
        match_0_tpc_label = models_labels[0][match_0]
        
        match_1 = match[1]
        match_1_tpc_desc = models_topics[1][match_1]
        match_1_tpc_label = models_labels[1][match_1]
        
        # Get the most representative document for each topic based on the retriever
        top_docs_retr = hermes_index.retrieve(match_0_tpc_label, 3)
        top_docs_retr = [el['original_document'] for el in top_docs_retr]
        most_repr_optim_s3 = [models_most_repr[0].iloc[match_0]["S3 most representative document #1"], models_most_repr[0].iloc[match_0]["S3 most representative document #2"], models_most_repr[0].iloc[match_0]["S3 most representative document #3"]]
        most_repr_optim_thetas = [models_most_repr[0].iloc[match_0]["thetas most representative document #1"], models_most_repr[0].iloc[match_0]["thetas most representative document #2"], models_most_repr[0].iloc[match_0]["thetas most representative document #3"]]
        
        top_docs_retr_1 = hermes_index.retrieve(match_1_tpc_label, 3)
        top_docs_retr_1 = [el['original_document'] for el in top_docs_retr_1]
        most_repr_non_optim_s3 = [models_most_repr[1].iloc[match_1]["S3 most representative document #1"], models_most_repr[1].iloc[match_1]["S3 most representative document #2"], models_most_repr[1].iloc[match_1]["S3 most representative document #3"]]
        most_repr_non_optim_thetas = [models_most_repr[1].iloc[match_1]["thetas most representative document #1"], models_most_repr[1].iloc[match_1]["thetas most representative document #2"], models_most_repr[1].iloc[match_1]["thetas most representative document #3"]]
        
        all_info.append(
            [
                match_0,
                match_0_tpc_desc,
                match_0_tpc_label,
                match_1,
                match_1_tpc_desc,
                match_1_tpc_label,
                top_docs_retr,
                most_repr_optim_s3,
                most_repr_optim_thetas,
                top_docs_retr_1,
                most_repr_non_optim_s3,
                most_repr_non_optim_thetas
            ]
        )
    
    df_results = pd.DataFrame(
        all_info,
        columns=[
            "Match 0",
            "Match 0 Topic Description",
            "Match 0 Topic Label",
            "Match 1",
            "Match 1 Topic Description",
            "Match 1 Topic Label",
            "Top Docs Optimized",
            "Most representative docs Optimized S3",
            "Most representative docs Optimized Thetas",
            "Top Docs Non-optimized",
            "Most representative docs Non-optimized S3",
            "Most representative docs Non-optimized Thetas"
        ]
    )

    # Initialize the pre-trained Sentence Transformer model
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    def average_cosine_similarity_embeddings(docs1, docs2):
        """
        Compute the average cosine similarity between two lists of documents using embeddings.

        Parameters:
        - docs1: List of documents (strings) from the first column.
        - docs2: List of documents (strings) from the second column.

        Returns:
        - avg_similarity: The average cosine similarity between the documents.
        """
        # Ensure that the lists are not empty and have the same length
        if not docs1 or not docs2 or len(docs1) != len(docs2):
            return None

        # Compute embeddings for both lists of documents
        embeddings1 = model.encode(docs1)
        embeddings2 = model.encode(docs2)

        # Compute cosine similarities between corresponding documents
        similarities = []
        for emb1, emb2 in zip(embeddings1, embeddings2):
            cos_sim = cosine_similarity([emb1], [emb2])[0][0]
            similarities.append(cos_sim)

        # Compute the average similarity
        avg_similarity = sum(similarities) / len(similarities)
        return avg_similarity

    # Define the pairs of columns you want to compare
    pairs = [
        ('Top Docs Optimized', 'Most representative docs Optimized S3'),
        ('Top Docs Optimized', 'Most representative docs Optimized Thetas'),
        ('Top Docs Non-optimized', 'Most representative docs Non-optimized S3'),
        ('Top Docs Non-optimized', 'Most representative docs Non-optimized Thetas')
    ]

    # Compute average cosine similarities and add them as new columns
    for col1, col2 in pairs:
        similarities = []
        for idx, row in df_results.iterrows():
            docs1 = row[col1]
            docs2 = row[col2]
            avg_sim = average_cosine_similarity_embeddings(docs1, docs2)
            similarities.append(avg_sim)
        # Add the results to the DataFrame
        df_results[f'Avg Cosine Similarity between {col1} and {col2}'] = similarities

    # Now, df contains the new columns with the average cosine similarities
    print(df_results.head())
    
    df_results.to_excel("results.xlsx")
    
    import pdb; pdb.set_trace()
    
if __name__ == "__main__":
   main() 