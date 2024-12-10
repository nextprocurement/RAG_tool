![Presentation](aux_files/HERMES_README.webp)
# HERMES Pipeline

HERMES tool was developed to improve the start of the art of topic modelling. HERMES integrates various text processing stages, including acronym detection and expansion, equivalence detection, and topic labelling. The pipeline leverages Large Language Models (LLMs) and linguistic preprocessing to produce semantically enriched outputs.

## Key Features
1. Acronym Detection and Expansion:
Uses LLMs to detect acronyms within text and expand them to their full forms.

2. Preprocessing:
Applies linguistic preprocessing steps such as lemmatization and optional embedding generation. This prepares the textual data for subsequent semantic analysis.

3. Equivalence Detection:
Identifies semantically equivalent words that have been previously clustered and standardizes them to a canonical form. This process ensures a cleaner and more consistent vocabulary.

4. Topic Modeling Training:
Trains various topic modeling approaches (e.g., MalletLda, Ctm, BERTopic) on the processed data. These models enable deeper semantic insights into the corpus.

## Command-line Arguments of Script **hermes_pipeline.py**

- `--llm_type`:  
  The type of large language model to use, e.g., `"llama"`, `"openai"`, `"mistral"`, etc.

- `--data_path`:  
  Path to the input data file.

- `--save_path`:  
  Path to save intermediate and final output files.

- `--mode`:  
  Pipeline mode (`"optimized"` or `"non-optimized"`).

- `--do_train`:  
  Indicates whether DSPy modules (e.g., for acronym detection/expansion and equivalences) should be trained.

- `--train_data_path`:  
  Path to training data for DSPy modules.

- `--context_window`, `--max_windows`, `--window_overlap`:  
  Parameters for windowing contexts in acronym detection/expansion.

- `--preproc_source`, `--lang`, `--spacy_model`:  
  Parameters for preprocessing steps (e.g., language, spaCy model).

- `--source_eq`, `--times_equiv`:  
  Parameters for equivalence detection (data source and number of iterations).

- `--num_topics`, `--num_iters`, `--model_type`, `--sample`:  
  Parameters for training topic models (number of topics, iterations, model type, and sample size).





