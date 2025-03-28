import pathlib
import logging
import pandas as pd
import re
import nltk
import yaml
import json
import os
import time
import logging
from time import sleep
from nltk.tokenize import sent_tokenize
from src.acronyms.acronym_detector import AcronymDetectorModule
from src.acronyms.acronym_expander import AcronymExpanderModule
from src.utils.vector_store_utils import Chunker
from src.acronyms.check_candidate import NERTextAnalyzer

def reorder_acronyms(acronyms_dict):
    acronyms = list(acronyms_dict.keys())
    expansions = list(acronyms_dict.values())
    assigned_expansions = [None] * len(acronyms)

    # Try matching with initial letter
    for i, acronym in enumerate(acronyms):
        for j, expansion in enumerate(expansions):
            if assigned_expansions[j] is None and expansion.lower().startswith(acronym[0].lower()):
                assigned_expansions[j] = acronym
                break

    # Matching for the highest number of initial letters
    for i, acronym in enumerate(acronyms):
        if acronym not in assigned_expansions:
            max_match = -1
            best_index = -1
            for j, expansion in enumerate(expansions):
                if assigned_expansions[j] is None:
                    matches = sum(1 for a, b in zip(acronym.lower(), expansion.lower()) if a == b)
                    if matches > max_match:
                        max_match = matches
                        best_index = j
            if best_index != -1:
                assigned_expansions[best_index] = acronym

    # Sorted dict
    corrected_dict = {assigned_expansions[i]: expansions[i] for i in range(len(acronyms)) if assigned_expansions[i] is not None}
    return corrected_dict

def substitute_acronyms(row, column_name, logger = None):
    text = row[column_name]
    try:
        if row['Expansions'] != "/":
            acronyms = row['Acronyms Detected(LLM)'].split(', ')
            expansions = row['Expansions'].split(', ')
            
            # Dict with acronyms and expansions
            acronyms_dict = dict(zip(acronyms, expansions))
            # Reorder acronyms
            if len(acronyms_dict) > 1:
                acronyms_dict = reorder_acronyms(acronyms_dict)

            for acronym, expansion in acronyms_dict.items():
                if expansion == '/':
                    continue
                pattern1 = re.compile(r'(?<!\w)' + re.escape(acronym) + r'(?!\w)', re.IGNORECASE)
                pattern2 = re.compile(r'\b' + r'\.?'.join(re.escape(char) for char in acronym) + r'\b', re.IGNORECASE)
                
                # Lambda function to substitute the acronym with its expansion
                text = pattern1.sub(lambda m: expansion, text)
                text = pattern2.sub(lambda m: expansion, text)        
            return text
        else:
            return text
    except Exception as e:
        logger.error(f"Error processing row {row.name}: {e}")
        return text

def generate_acronym_expansion_json(file_path, output_dir):
    """
    Generate a JSON file containing the acronyms and their expansions from an Excel file.
    df: Dataframe containing the acronyms and expansions, with columns
                'Acronyms Detected(LLM)' and 'Expansions'.
    output_dir: Path to the directory where the JSON file will be saved.
    """
    #Load the Excel file into a DataFrame
    print(f"Cargando archivo desde: {file_path}")

    # Obtain the file extension
    file_extension = os.path.splitext(file_path)[1]

    if file_extension == '.xlsx' or file_extension == '.xls':
        df_out = pd.read_excel(file_path)
    elif file_extension == '.parquet':
        df_out = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Formato de archivo no soportado: {file_extension}")

    acronym_expansion_dict = {}
    for index, row in df_out.iterrows():
        # Obtain the acronyms and expansions from the DataFrame
        acronyms = row['Acronyms Detected(LLM)'].split(',')
        expansions = row['Expansions'].split(',')

        # Clean and format the acronyms and expansions
        acronyms = [acronym.strip() for acronym in acronyms]
        expansions = [expansion.strip().replace(' ', '_') for expansion in expansions]

        # Associate each acronym with its expansion
        for acronym, expansion in zip(acronyms, expansions):
            # Add the acronym and expansion to the dictionary
            if expansion != '/':
                acronym_expansion_dict[acronym] = expansion

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    json_name = f"{file_name}_equivalences.json"

    json_data = {
        "name": file_name,
        "description": "",
        "valid_for": "acronyms",
        "visibility": "Public",
        "wordlist": [f"{key}:{value}" for key, value in acronym_expansion_dict.items()]
    }
    # Path to save the JSON file
    output_path = os.path.join(output_dir, json_name)
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(json_data, file, ensure_ascii=False, indent=4)
    print("JSON file saved correctly.")
    
def process_dataframe(
    path,
    config,
    action,
    chunker=None, 
    acronym_detector=None, 
    acronym_expander=None,
    context_window=3000, 
    max_windows=100, 
    window_overlap=0.1,
    logger=None,
    save_directory="processed_chunks"
):
    """
    Processes a DataFrame of text to detect and expand acronyms based on the action specified.
    
    Parameters:
    - path: str or Path, path to the Excel file containing the data.
    - config: dict, configuration loaded from the YAML file.
    - action: str, action to perform: "detect", "expand", or "both".
    - chunker: instance of Chunker, if not provided, a new one is created with the specified parameters.
    - acronym_detector: instance of AcronymDetectorModule, if not provided, a new one is created.
    - acronym_expander: instance of AcronymExpanderModule, used to expand detected acronyms.
    - context_window: int, size of the context window for the chunker.
    - max_windows: int, maximum number of windows the chunker can generate.
    - window_overlap: float, percentage of overlap between windows generated by the chunker.
    - save_directory: str, directory to save the processed chunks.

    Returns:
    - df: Processed DataFrame with detected and/or expanded acronyms.
    """
    # Checking if the save_directory exists
    os.makedirs(save_directory, exist_ok=True)
    
    # Initialize NERTextAnalyzer
    ner_analyzer = NERTextAnalyzer()
    # Obtain the column name from the configuration file
    column_name = config['data_column_name']
    print("Column name: ", column_name)

    # Load the DataFrame based on the file extension
    if path.endswith('.xlsx'):
        df = pd.read_excel(path).copy()
    elif path.endswith('.parquet'):
        df = pd.read_parquet(path).copy()
    else:
        raise ValueError("Unsupported file format. Please provide a .xlsx or .parquet file.")

    if chunker is None:
        chunker = Chunker(context_window=context_window, max_windows=max_windows, window_overlap=window_overlap)
    print("El archivo para procesar y detectar/expandir acr es", df)
    # Sus cols son 'objective', 'Acronyms Detected(LLM)', 'Expansions', 'text_substituted', 'id_tm'
    # Add columns to store the detected acronyms and their expansions
    if action in ["detect", "both"]:
        df['Acronyms Detected(LLM)'] = ''
    if action in ["expand", "both"]:
        df['Expansions'] = ''
        df['text_substituted'] = ''

    rows_per_chunk = 40000
    processed_rows = 0
    # Create temporal dataframe to store the processed rows
    temp_df = pd.DataFrame(columns=df.columns)
    
    # Iterate over each row in the DataFrame
    for identifier, row in df.iterrows():
        text = row[column_name]
        print(f"PROCESSING ROW WITH TEXT: {text}")
        if not text:
            logger.error("La columna especificada no contiene texto. Verifica el archivo YAML.")
            continue
        
        detected_acronyms = set()  # Initialize detected_acronyms as an empty set

        # Perform detection if action is "detect" or "both"
        if action in ["detect", "both"] and acronym_detector:
            for id_chunk, chunk in chunker(text):
                prediction = acronym_detector.forward(chunk)
                acronyms = clean_acronyms(prediction.ACRONYMS)
                acronyms_list = acronyms.lower().split(',')
                #print("ACRONYMS DETECTED DETECTOR MODULE:", acronyms_list)

                if not isinstance(detected_acronyms, set):
                    # Initialize the set of detected acronyms if it is not a set
                    detected_acronyms = set()

                if isinstance(acronyms_list, str):
                    acronyms_list = [acronyms_list]

                # Filter detected acronyms
                detected_acronyms.update(acronym.strip() for acronym in acronyms_list)
                
                if not detected_acronyms or '/' in detected_acronyms:
                    #print(f"No acronyms in row {identifier}, continue the loop.")
                    df.at[identifier, 'Acronyms Detected(LLM)'] = '/'
                    continue
                    
                # Apply filters to the detected acronyms
                detected_acronyms = filter_split_characters(detected_acronyms)
                #print(f"ACRONYMS AFTER filter_split_characters: {detected_acronyms}")
                detected_acronyms = filter_items_and_acronyms(detected_acronyms)
                #print(f"ACRONYMS AFTER filter_items_and_acronyms: {detected_acronyms}")
                detected_acronyms = filter_companies(detected_acronyms)
                #print(f"ACRONYMS AFTER filter_companies: {detected_acronyms}")
                detected_acronyms = filter_acronyms_in_text(text, detected_acronyms)
                #print(f"ACRONYMS AFTER filter_acronyms_in_text: {detected_acronyms}")

                if not detected_acronyms or '/' in detected_acronyms:
                    #print(f"No acronyms AFTER FILTERING in row {identifier}, continue the loop.")
                    df.at[identifier, 'Acronyms Detected(LLM)'] = '/'
                    continue
                
                # Convert set to list for NER analysis
                detected_acronyms_list = list(detected_acronyms)
                #print(f"ACRONYMS BEFORE NER FILTERING: {detected_acronyms_list}")
                
                # Analyze acronyms using NERTextAnalyzer
                detected_acronyms = ner_analyzer.analyze_text(text, detected_acronyms_list)
                #print(f"Acronyms after ner: {detected_acronyms}")
                
                if not detected_acronyms or '/' in detected_acronyms:
                    #print(f"No acronyms AFTER FILTERING in row {identifier}, continue the loop.")
                    df.at[identifier, 'Acronyms Detected(LLM)'] = '/'
                    continue
                                
                # Update the DataFrame with the detected acronyms
                df.at[identifier, 'Acronyms Detected(LLM)'] = ', '.join(detected_acronyms)

        # Perform expansion if action is "expand" or "both"
        if action in ["expand", "both"] and acronym_expander:
            if action == "expand":
                if 'Acronyms Detected(LLM)' not in df.columns:
                    logger.error("Column 'Acronyms Detected(LLM)' does not exists in DataFrame. Cannot apply expansion before detection!.")
                    continue
            
            acronyms_detected = df.at[identifier, 'Acronyms Detected(LLM)']
            #print(f"ACRONYMS DETECTED IN ROW {identifier}: {acronyms_detected}")

            # Skip the row if acronyms are not detected or contain '/' or are empty
            if acronyms_detected in ['/', '']:
                logger.error(f"No acronyms detected in the row {identifier}. No expansion its needed.")
                df.at[identifier, 'Expansions'] = '/'
                continue

            detected_acronyms = set(acronym.strip() for acronym in acronyms_detected.split(','))
            expansions = []
            
            # Iterate through each acronym in the detected list
            for acronym in detected_acronyms:
                try:
                    # Expand the acronym using the forward method of AcronymExpanderModule
                    expansion_response = acronym_expander.forward(texto=text, acronimo=acronym)
                    expansion = expansion_response.EXPANSION
                    #print(f"EXPANSION FOR ACRONYM {acronym}: {expansion}")
                    expansions.append(expansion)
                      
                except Exception as e:
                    logger.error(f"Error expanding {acronym} in row {identifier}: {e}")

            # Update the DataFrame with the expansions for each row
            df.at[identifier, 'Expansions'] = ', '.join(expansions)

        # Add the processed row to the temporary DataFrame
        temp_df = pd.concat([temp_df, df.loc[[identifier]]], ignore_index=True)
        processed_rows += 1

        # Each 40000 rows save the processed chunk
        if processed_rows % rows_per_chunk == 0:
            new_chunk_size = processed_rows  
            new_chunk_file = os.path.join(save_directory, f"processed_rows_{new_chunk_size}.parquet")
            
            # Verify if there is a previous chunk
            previous_chunk_size = processed_rows - rows_per_chunk
            if previous_chunk_size > 0:
                previous_chunk_file = os.path.join(save_directory, f"processed_rows_{previous_chunk_size}.parquet")
                
                if os.path.exists(previous_chunk_file):
                    # Check if the previous chunk exists
                    old_df = pd.read_parquet(previous_chunk_file)
                    combined_df = pd.concat([old_df, temp_df], ignore_index=True)
                    
                    # Guardar archivo acumulado con las filas anteriores + las nuevas
                    combined_df.to_parquet(new_chunk_file)
                    
                    # Eliminar el chunk anterior
                    os.remove(previous_chunk_file)
                    logger.info(f"Guardado archivo acumulado con {new_chunk_size} filas y eliminado chunk anterior: {previous_chunk_file}")
                else:
                    # If the previous chunk does not exist, save only the current one
                    temp_df.to_parquet(new_chunk_file)
                    logger.warning(f"No se encontró el chunk anterior. Se guarda solo el actual con {new_chunk_size} filas.")
            else:
                # First chunk to process
                temp_df.to_parquet(new_chunk_file)
                logger.info(f"Guardado primer chunk con {new_chunk_size} filas en {new_chunk_file}")
            
            # Reiniciar temp_df para siguiente bloque de filas
            temp_df = pd.DataFrame(columns=df.columns)

    # Final save of the processed chunk
    if not temp_df.empty:
        new_chunk_size = processed_rows
        new_chunk_file = os.path.join(save_directory, f"processed_rows_{new_chunk_size}.parquet")

        # Verify if there is a previous chunk
        previous_chunk_size = (processed_rows // rows_per_chunk) * rows_per_chunk
        if previous_chunk_size > 0:
            previous_chunk_file = os.path.join(save_directory, f"processed_rows_{previous_chunk_size}.parquet")
            if os.path.exists(previous_chunk_file):
                # Concatenate the previous chunk with the temporary DataFrame
                old_df = pd.read_parquet(previous_chunk_file)
                combined_df = pd.concat([old_df, temp_df], ignore_index=True)
                combined_df.to_parquet(new_chunk_file)
                # Remove the previous chunk
                os.remove(previous_chunk_file)
                logger.info(f"Guardado final acumulado con {new_chunk_size} filas y eliminado chunk anterior.")
            else:
                # If the previous chunk does not exist, save only the current one
                temp_df.to_parquet(new_chunk_file)
                logger.warning(f"No se encontró el chunk anterior. Se guarda el final con {new_chunk_size} filas.")
        else:
            temp_df.to_parquet(new_chunk_file)
            logger.info(f"Guardado final con {new_chunk_size} filas en {new_chunk_file}")
    # Return only the relevant columns based on the action
    if action == "detect":
        return df[[column_name, 'Acronyms Detected(LLM)']]
    elif action in ["both", "expand"]:
        df['text_substituted'] = df.apply(substitute_acronyms, axis=1, args=(column_name,logger))
        df['id_tm'] = range(len(df))
        df = df.applymap(lambda x: x.replace('\n', ' ').replace('\r', ' ') if isinstance(x, str) else x)
        
        return df[[column_name,'Acronyms Detected(LLM)','Expansions', 'text_substituted', 'id_tm']]
    else:
        # Ensure that acronyms have been detected before expanding
        if 'Acronyms Detected(LLM)' not in df.columns or df['Acronyms Detected(LLM)'].isnull().all():
            logger.error("No se puede aplicar la expansión porque no se detectaron acrónimos.")
            raise ValueError("La expansión no puede aplicarse sin detección previa de acrónimos.")
        return df[[column_name,'Acronyms Detected(LLM)','Expansions']]





def filter_split_characters(acronyms):
    """
    Processes a list of acronyms. If an acronym contains parts separated by spaces
    and one of the parts is completely numeric, it retains only the alphabetic parts.
    """
    filtered_acronyms = []
    for acronym in acronyms:
        parts = acronym.split()
        # Check if there is at least one numeric part
        if any(part.isdigit() for part in parts):
            # Retain only alphabetic parts
            filtered_acronyms.append(' '.join(part for part in parts if not part.isdigit()))
        else:
            filtered_acronyms.append(acronym)
              
    return filtered_acronyms

def filter_acronyms_in_text(text, acronyms):
    '''
    Filter acronyms based on their presence in the text. Taking into account acronyms enclosed in parentheses.
    '''
    text_lower = text.lower()
    # Find all acronyms enclosed in parentheses
    acronyms_in_parentheses = re.findall(r'\(([^)]+)\)', text_lower)
    # Flatten the list of acronyms found in parentheses and convert to lowercase
    acronyms_in_parentheses = [acronym.lower() for group in acronyms_in_parentheses for acronym in group.split()]

    filtered_acronyms = []
    # Iterate through each acronym and check if it exists in the text
    for acronym in acronyms:
        acronym_lower = acronym.lower()
        # Define patterns to match the acronym with potential surrounding punctuation
        patterns = [
            rf'\b{re.escape(acronym_lower)}\b',  # Exact match
            rf'\b{re.escape(acronym_lower)}[.,;:!?]',  # Acronym followed by punctuation
            rf'[.,;:!?]{re.escape(acronym_lower)}\b',  # Acronym preceded by punctuation
            rf'{re.escape(acronym_lower)}'  # General match anywhere in the text
        ]
        # Check if the acronym appears in the text or in acronyms found in parentheses
        if any(re.search(pattern, text_lower) for pattern in patterns) or acronym_lower in acronyms_in_parentheses:
            filtered_acronyms.append(acronym)

    return filtered_acronyms if filtered_acronyms else '/'

def filter_companies(acronyms):
    """
    Filters out acronyms that contain any variant of 'SL', 'SLU', or 'SA'.
    """
    pattern = re.compile(r'\b(?:S\.?L\.?(?:U\.?)?|S\.?A\.?)\b', re.IGNORECASE)
    filtered_acronyms = [acronym for acronym in acronyms if not pattern.search(acronym)]
    return filtered_acronyms

def filter_items_and_acronyms(items):
    """
    Filters items from a list based on the following criteria:
    - The item must be 14 characters or fewer.
    - The item must contain only one word.
    - The item must have three or fewer digits.
    - The item must be longer than one character.
    - The item must not be a number.
    """
    filtered_items = [
        item for item in items
        if len(item) <= 14
        and len(item.split()) == 1
        and sum(c.isdigit() for c in item) <= 3
        and len(item) > 1
        and not item.isdigit()
    ]
    return filtered_items

def disambiguation_in_text(text, acronym, expansion):
    """
    This function locates all occurrences of the acronym in the text and replaces them with the expansion.
    
    text: The text in which the replacement should be made.
    acronym: The acronym to search for in the text.
    expansion: The expansion that will replace the acronym in the text.
    """
    # Regex pattern to find the acronym
    pattern = re.compile(r'\b' + re.escape(acronym) + r'\b')
    
    # Replace all occurrences of the acronym with the expansion
    result_text = pattern.sub(expansion, text)
    
    return result_text 

def extract_passages(text, acronym):
        """
        Extract the two preceding and two following sentences after the first appearance of the acronym in
        the text.
        """
        sentences = sent_tokenize(text)
        for i, sentence in enumerate(sentences):
            if acronym.lower() in sentence.lower():
                start = max(i - 2, 0)
                end = min(i + 3, len(sentences))
                return ' '.join(sentences[start:end])
        return None

def clean_acronyms(text):
    """
    Remove '(..text..)' inside parenthesis. Useful to clean LLM extra information.
    """
    cleaned_text = re.sub(r'\s*\([^)]*\)', '', text)
    return cleaned_text.strip()

def init_logger(config: dict) -> logging.Logger:
    """
    Initialize a logger based on the provided configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing logger settings.

    Returns
    -------
    logging.Logger
        The initialized logger.
    """
    name = config.get("logger_name", "app-log")
    log_level = getattr(logging, config.get("log_level", "INFO").upper())
    dir_logger = pathlib.Path(config.get("dir_logger", "logs/app.log"))

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create path_logs dir if it does not exist
    dir_logger.parent.mkdir(parents=True, exist_ok=True)

    # Create handlers based on config
    if config.get("file_log", True):
        file_handler = logging.FileHandler(dir_logger)
        file_handler.setLevel(log_level)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    if config.get("console_log", True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    return logger