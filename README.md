# RAG Tool - Acronym Detection Module

This project enables acronym detection using a detection module that can run with or without prior training. Currently, only the detect module is available. It is available for parquet and excel datasets. Given the name of the column where text its contained -> create an extension file with same name plus suffix '_out' with new column where acronyms are located.

## Usage Instructions

### 1. Run the Detection Module Without Training (Load Previously Trained Model)

To run the detection module without training a new model, you can load a previously trained model and provide the input data file.

```bash
python3 main.py --action detect --data_path /export/usuarios_ml4ds/cggamella/RAG_tool/files/anotacion_manual/fam/df_esp_100_200.xlsx
```

### 2. Run the Detection Module With Training (Load and Train a New Model)

If you want to train the model before running detection, set the --do_train parameter to enable training.

```bash
python3 main.py --action detect --data_path /export/usuarios_ml4ds/cggamella/RAG_tool/files/anotacion_manual/fam/df_esp_100_200.xlsx --do_train
```

### 3. Run the Detection Module Without Training for CPV-45

To run detection without training on a specific .parquet file, use the following command:

```bash
python3 main.py --action detect --data_path /export/usuarios_ml4ds/cggamella/RAG_tool/files/anotacion_manual/fam/datos_modelo_es_Mallet_df_merged_14_topics_45_ENTREGABLE.parquet
```





