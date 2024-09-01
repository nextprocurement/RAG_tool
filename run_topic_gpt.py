import argparse, os

from src.topic_modeling.topic_model import TopicGPTModel
from src.utils.tools import load_api_key

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--load_data_path",
        help="Path to the preprocessed data.",
        type=str, 
        default='/export/usuarios_ml4ds/lbartolome/Repos/umd/bass_data/trained/Data/education/education_eval.json',
        required=False)
    argparser.add_argument(
        "--model_path",
        help="The model path to save the trained models",
        type=str, 
        default='/export/usuarios_ml4ds/lbartolome/Repos/umd/bass_data/trained/Models/education_eval', 
        required=False)
    argparser.add_argument(
        "--manager",
        help="Manager running the script",
        type=str, 
        default="lorena"
    )
    argparser.add_argument(
        "--sample",
        help="how many documents to run",
        type=int, 
        required=False
        #default=100#100
    )
    argparser.add_argument(
        "--do_second_level",
        help="Whether to generate second-level topics.",
        type=bool, 
        default=True
    )
    
    args = argparser.parse_args()
    
    # Load OpenAI Api key
    api_key = load_api_key('key.txt', args.manager)
    os.environ['OPENAI_API_KEY'] = api_key
    # print(api_key)
    # exit()
    
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