import argparse

from src.topicmodeling.topic_model import (BERTopicModel, CtmModel,
                                            MalletLdaModel, TomotopyLdaModel)


def create_model(model_name, **kwargs):
    # Map model names to corresponding classes
    model_mapping = {
        'MalletLda': MalletLdaModel,
        'TomotopyLda': TomotopyLdaModel,
        'Ctm': CtmModel,
        'BERTopic': BERTopicModel,
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
        type=int, default=1, required=False)
    argparser.add_argument(
        "--model_type",
        help="type of the model, MalletLda, TomotopyLda, Ctm, BERTopic",
        type=str, default='MalletLda', required=False)
    argparser.add_argument(
        "--load_data_path",
        help="Path to the preprocessed data.",
        type=str, default='data/source/cordis_preprocessed.json',
        required=False)
    argparser.add_argument(
        "--model_path",
        help="The model path to save the trained models",
        type=str, default='data/models/test_model', required=False)

    args = argparser.parse_args()

    # Create a dictionary of parameters
    list_skip = ['model_type']

    if args.model_type == 'BERTopic':
        list_skip += ['num_iters']

    params = {k: v for k, v in vars(args).items()
              if v is not None and k not in list_skip}

    # Create a model instance of type args.model_type
    model = create_model(args.model_type, **params)

    # Train the model
    model.train()

    pass


if __name__ == "__main__":
    main()