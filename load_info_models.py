import argparse

from src.topicmodeling.utils import create_model

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--model_type",
        help="num of topics",
        type=str,
        default='MalletLda',
        required=False)
    argparser.add_argument(
        "--load_data_path",
        help="the path of the training data",
        type=str,
        default='data/all_extracted_12aug_es_11_objectives_embeddings.parquet', required=False)
    argparser.add_argument(
        "--load_model_folder",
        help="the folder of the trained models",
        type=str,
        default='data/models/TopicGPT_83',
        required=False)

    args = argparser.parse_args()

    tm_params = {
        'load_data_path': args.load_data_path,
        'model_path': 'FOLDER_OF_TRAINED_TOPIC_MODEL'
    }

    params_inference = tm_params
    params_inference['load_model'] = True
    params_inference['model_path'] = args.load_model_folder

    model = create_model(args.model_type, **params_inference)
    topics = model.print_topics()

    for i, topic in enumerate(topics):
        print("Topic #", i)
        print(topics[topic])

    print('-'*100)


if __name__ == "__main__":
    main()
