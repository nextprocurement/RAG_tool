# RAG_TOOL

## CLEANING-ENRICHMENT

This directory houses modules that implement a Natural Language Processing (NLP) pipeline.

The core logic of the pipeline is encapsulated in the `NLPpipeline` class, defined in `nlp_pipeline.py`. Our approach to designing this pipeline is modular, allowing users to customize their preprocessing workflow by specifying configuration parameters when creating an instance of the `NLPpipeline` class. This flexibility empowers users to select which processing steps to apply and in what sequence.

The NLP pipeline comprises the following steps:

- **Language Detection**
- **Acronyms Extraction**
- **Lemmatization**
- **N-grams Extraction**
- **Contextual Embeddings Extraction**
- **Named Entity Recognition (NER) Extraction**
- **Specific NER Extraction**

> Although each method is designed to be callable independently, it's important to note that certain methods are interdependent. For instance, `get_ngrams` and `get_ner_generic` rely on the lemmas extracted from the text column. Therefore, it is imperative that lemmas are computed beforehand.

## Configuration

The file `config/config.yaml` serves as the configuration hub for the NLP pipeline. It consists of two sections:

1. **Pipeline Selection (`pipe`):**
   Users can tailor the pipeline's behavior by choosing which elements to activate. To exclude a step, simply comment out the corresponding line.

   ```yaml
   pipe:
     - lang_id        # Identifies the language of the text
     - acronyms       # Identifies the acronyms of the text
     - lemmas         # Creates lemmas from the text
     - ngrams         # Generates n-grams from the text
     - embeddings     # Generates BERT embeddings
     - ner_generic    # Identifies generic named entities (e.g. person, location, organization)
     - ner_specific   # Identifies specific named entities (e.g. Apple, Microsoft, New York)
   ```

2. **Module Parameters (`options_pipe`):**
   This section includes parameter options for the various modules:
   - **spaCy_model**: SpaCy language model for text processing.

   - **lang**: Language of the text being processed.

   - **valid_POS**: Valid parts of speech tags for filtering.

   - **gensim_phraser_min_count**: Minimum count for Gensim phrase detection.

   - **gensim_phraser_threshold**: Threshold for merging phrases in Gensim.

   - **sentence_transformer_model**: Pre-trained Sentence Transformer model.

   - **batch_size_embeddings**: Batch size for embedding generation.

   - **aggregate_embeddings**: When enabled, this feature calculates embeddings for large texts by truncating them into chunks matching the model's context size, then averages these chunk embeddings.

   - **use_gpu**: Whether to use GPU.
  
3. **Default Spacy models for each language (`langs_dft`):**
   - **spaCy_model_dft_en**: Default SpaCy language model for text processing in English.
   - **spaCy_model_dft_es**: Default SpaCy language model for text processing in Spanish.


### Module-specific Configuration

#### Specific NER Extraction

The Specific NER Extraction function uses SpaCy's "spacy.NER.v3" task to identify specific named entities such as "DATE" (for dates) and "DRUG" (for drugs). It can be tailored to recognize other named entity types with a detailed description and positive/negative examples for training.

To configure entity extraction, refer to `cleaning_enrichment/src/ner_specific_extractor/config.cfg`. Including examples in `cleaning_enrichment/src/ner_specific_extractor/examples.json` improves module performance.

For detailed customization guidance, consult the module's documentation.

## Entrypoint

The `pipe.py` file serves as the main entry point for the NLP pipeline. Within this file, the `cal_element_pipe` function orchestrates the invocation of each selected module based on the user's configuration file. These configurations are then passed to the instantiation of the class.