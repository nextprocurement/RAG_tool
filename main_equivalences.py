from src.equivalences.equivalences_generator import HermesEquivalencesGenerator


def main():
   eg = HermesEquivalencesGenerator(
        use_optimized = True,
        do_train = True,
   )
   
   path_model = "/export/usuarios_ml4ds/lbartolome/Repos/repos_con_carlos/RAG_tool/data/models/cpv45/MalletLda_100"
   path_vocab = "/export/usuarios_ml4ds/lbartolome/Repos/repos_con_carlos/RAG_tool/data/models/cpv45/MalletLda_100/vocabulary.txt"
   
   eg.generate_equivalences(
        source = "tm",#"vocabulary"
        path_to_source = path_model,
        path_save = "/export/usuarios_ml4ds/lbartolome/Repos/repos_con_carlos/RAG_tool/src/topicmodeling/data/equivalences/cpv45_equivalences_tm_trained.json",
        model_type = "MalletLda",
        language = "spanish",
   )


if __name__ == "__main__":
    main()