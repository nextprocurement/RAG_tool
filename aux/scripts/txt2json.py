import json
import pathlib

path_stops = "/export/usuarios_ml4ds/lbartolome/Repos/repos_con_carlos/RAG_tool/data/out_pliegos/optimized/stops_txt"

file_count = sum(1 for file in pathlib.Path(path_stops).rglob('*') if file.is_file())

for el, path in enumerate(pathlib.Path(path_stops).rglob('*')):
    print("*" * 50)
    print(f"-- -- Processing file {el+1} / {file_count}: {path}")
    print("*" * 50)
    
    words = []
    with open(path, "r") as file:
        for line in file:
            word = line.strip()
            words.append(word)
    
    json_data = {
        "name": pathlib.Path(path_stops).stem,
        "description": "",
        "valid_for": "stopwords",
        "visibility": "Public",
        "wordlist": words
    }
    
    path_save = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/repos_con_carlos/RAG_tool/data/out_pliegos/optimized/stops_json") / f"{path.stem}.json"
    with open(path_save, 'w') as json_file:
        json.dump(json_data, json_file, indent=4, ensure_ascii=False)