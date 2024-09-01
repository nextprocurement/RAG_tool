import argparse
import os
import pathlib

from dotenv import load_dotenv

from src.topicmodeling.topic_model import (BERTopicModel, CtmModel,
                                            MalletLdaModel, TopicGPTModel)


def create_model(model_name, **kwargs):
    # Map model names to corresponding classes
    model_mapping = {
        'MalletLda': MalletLdaModel,
        'Ctm': CtmModel,
        'BERTopic': BERTopicModel,
        'TopicGPT': TopicGPTModel
    }

    # Retrieve the class based on the model name
    model_class = model_mapping.get(model_name)

    # Check if the model name is valid
    if model_class is None:
        raise ValueError(f"Invalid model name: {model_name}")

    # Create an instance of the model class
    model_instance = model_class(**kwargs)

    return model_instance


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--num_topics",
        help="Number of topics",
        type=int, default=83, required=False)
    argparser.add_argument(
        "--num_iters",
        help="Number of iterations",
        type=int, default=1000, required=False)
    argparser.add_argument(
        "--model_type",
        help="type of the model, MalletLda, Ctm, BERTopic, all",
        type=str, default='all', required=False)
    argparser.add_argument(
        "--load_data_path",
        help="Path to the preprocessed data.",
        type=str, default='data/all_extracted_12aug_es_11_objectives_embeddings.parquet',
        required=False)
    argparser.add_argument(
        "--model_path",
        help="Path to save the trained models",
        type=str, default='data/models', required=False)
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
    
    if args.model_type == 'all':
        for model_type in ['TopicGPT']:#'MalletLda', 'Ctm', 'BERTopic'
            
            model_path = pathlib.Path(args.model_path) / f"{model_type}_{args.num_topics}"
            print(f"-- -- Training model of type {model_type} at {model_path}...")
            
            list_skip = ['model_type']
            if model_type == 'BERTopic':
                list_skip += ['num_iters']
            
            if model_type in ['MalletLda', 'Ctm', 'BERTopic']:
                list_skip += ['do_second_level', 'sample']
            
            if model_type == 'TopicGPT':
                path_env = pathlib.Path(".env")
                load_dotenv(path_env)
                api_key = os.getenv("OPENAI_API_KEY")
                os.environ["OPENAI_API_KEY"] = api_key
                
                list_skip += ['num_iters']
                
            params = {k: v for k, v in vars(args).items() if v is not None and k not in list_skip}
            params["model_path"] = model_path
            if model_type == 'TopicGPT':
                params["api_key"] = api_key

            # Create a model instance of type args.model_type
            model = create_model(model_type, **params)

            # Train the model
            model.train()


if __name__ == "__main__":
    main()