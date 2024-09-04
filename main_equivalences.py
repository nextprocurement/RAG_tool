from src.equivalences.equivalences_generator import HermesEquivalencesGenerator


def main():
   eg = HermesEquivalencesGenerator(
        use_optimized = False,
        do_train = False,
   )
   
   path_model = "/export/usuarios_ml4ds/lbartolome/Repos/repos_con_carlos/RAG_tool/data/models/cpv45/MalletLda_100"
   
   eg.generate_equivalences(
        source = "tm",
        path_to_source = path_model,
        model_type = "MalletLda",
        language = "spanish",
   )


if __name__ == "__main__":
    main()