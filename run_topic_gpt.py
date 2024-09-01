import argparse, os
import pathlib
from dotenv import load_dotenv
from src.topicmodeling.topic_model import TopicGPTModel

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--load_data_path",
        help="Path to the preprocessed data.",
        type=str, 
        default='data/all_extracted_12aug_es_11_objectives_embeddings.parquet',
        required=False)
    argparser.add_argument(
        "--model_path",
        help="The model path to save the trained models",
        type=str, 
        default='data/models/topicgpt', 
        required=False)
    argparser.add_argument(
        "--sample",
        help="how many documents to run",
        type=int, 
        required=False,
        default=10
    )
    argparser.add_argument(
        "--do_second_level",
        help="Whether to generate second-level topics.",
        type=bool, 
        default=True
    )
    
    args = argparser.parse_args()
    
    # Load OpenAI Api key
    path_env = pathlib.Path(".env")
    load_dotenv(path_env)
    api_key = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = api_key
    
    model = TopicGPTModel(
        api_key = api_key,
        load_data_path = args.load_data_path,
        load_model = False,
        model_path = args.model_path,
        sample = args.sample,
        do_second_level = args.do_second_level
    )

    # exit()
    
    model.train()
    
if __name__ == "__main__":
    main()