import argparse
import os
import pathlib

from dotenv import load_dotenv
import dspy
from src.acronyms.acronym_expander import HermesAcronymExpander


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acronym Expander")
    parser.add_argument("--llm_type", type=str, default="llama",
                        help="Type of language model to use")
    parser.add_argument("--data_path", type=str,
                        help="Path to data file",
                        default='/export/usuarios_ml4ds/cggamella/RAG_tool/acronyms_paper.xlsx')

    args = parser.parse_args()

    # Load llm ton use
    if args.llm_type == "llama":
        lm = dspy.HFClientTGI(
            model="meta-llama/meta-llama-3-8b-instruct", port=8090, url="http://127.0.0.0")
    elif args.llm_type == "gpt":
        # @TODO: Add to config
        path_env = pathlib.Path(
            "/export/usuarios_ml4ds/cggamella/NP-Search-Tool/.env")
        load_dotenv(path_env)
        api_key = os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = api_key

        lm = dspy.OpenAI(model="gpt-3.5-turbo")
    else:
        raise ValueError(f"Invalid LLM type: {args.llm_type}")

    dspy.settings.configure(lm=lm, temperature=0)
    
    ########################################
    # Get AcronymExpander
    ########################################
    expander = HermesAcronymExpander(
        do_train=True, data_path=args.data_path)
    
    # Now it can be used as expander.module(...)
    expander.module(texto="Quiero participar en una ONG porque quiero ser caritativa", acronimo="ONG")
