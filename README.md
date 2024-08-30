# In order to run detect module without train (load previous trained model)
python3 main.py --action detect --data_path /export/usuarios_ml4ds/cggamella/RAG_tool/files/anotacion_manual/fam/df_esp_100_200.xlsx

# In order to run detect module with TRAIN (load previous trained model) CONFIGURE do_train
python3 main.py --action detect --data_path /export/usuarios_ml4ds/cggamella/RAG_tool/files/anotacion_manual/fam/df_esp_100_200.xlsx --do_train



