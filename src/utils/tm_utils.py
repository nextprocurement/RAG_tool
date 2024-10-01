import os
import pathlib

from dotenv import load_dotenv
from src.topicmodeling.topic_model import (BERTopicModel, CtmModel, MalletLdaModel, TopicGPTModel)

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

def train_model(
    model_path,
    model_type,
    num_topics,
    further_proc,
    logger,
    env,
    args,
    stw_path=None,
    eq_path=None
    ):
    
    model_path = pathlib.Path(model_path) / f"{model_type}_{num_topics}"
    logger.info(f"-- -- Training model of type {model_type} at {model_path}...")
    
    list_skip = ['model_type','further_proc']
    if model_type == 'BERTopic':
        list_skip += ['num_iters']
    
    if model_type in ['MalletLda', 'Ctm', 'BERTopic']:
        list_skip += ['do_second_level', 'sample']
    
    if model_type == 'TopicGPT':
        load_dotenv(env)
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
    model.train(further_proc=further_proc, stops_path=stw_path, eqs_path=eq_path)
    
    return model