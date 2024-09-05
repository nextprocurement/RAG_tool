import spacy
from src.acronyms.acronym_expander import LanguageDetectorModule
#from acronym_expander import LanguageDetectorModule
import logging

class NERTextAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("es_core_news_md")  

    def load_model(self, language):
        """
        Load the appropriate spaCy model based on detected language and print the action.
        """
        if language == "SPANISH":
            self.nlp = spacy.load("es_core_news_md")
            print("Model 'es_core_news_md' loaded for Spanish language.")
        elif language == "ENGLISH":
            self.nlp = spacy.load("en_core_web_md")
            print("Model 'en_core_web_md' loaded for English language.")
        elif language == "CATALAN":
            self.nlp = spacy.load("ca_core_news_md")
            print("Model 'ca_core_news_md' loaded for Catalan language.")
        else:
            self.nlp = spacy.load("es_core_news_md")
            print(f"Language detected: {language}. Unsupported language, loading Spanish model by default.")

    def analyze_text(self, text, detected_acronyms):
        """
        Analyze the text and filter detected acronyms based on POS and NER criteria.

        Parameters:
        - text (str): The text to analyze.
        - detected_acronyms (list): A list of acronyms detected in the text.

        Returns:
        - List of filtered acronyms after applying POS and NER filters.
        """
        # Detect language of the text and load the corresponding spaCy model
        language_detector = LanguageDetectorModule()
        detected_language = language_detector.detect_language(text)
        print("Detected language is:", detected_language)
        # Load the appropriate model based on the detected language
        self.load_model(detected_language)

        # Process text with the spaCy model
        doc = self.nlp(text)
        # Convert detected acronyms to lowercase for consistent comparisons
        detected_acronyms = {acronym.lower() for acronym in detected_acronyms}
        print("Acrónimos detectados:", detected_acronyms)

        # Separar cada acrónimo en sus palabras componentes
        acronym_words = {}
        for acronym in detected_acronyms:
            words = acronym.split()
            acronym_words[acronym] = words
        print("\nPalabras separadas de los acrónimos:", acronym_words)

        # POS and NER exclusion criteria
        pos_to_exclude = {"ADJ", "PUNCT", "VERB"}
        ner_labels_to_exclude = {"GPE", "TIME", "MONEY", "FAC"}

        # Filter acronyms based on POS, sometimes they are composed of more than 1 word
        acronyms_to_keep = set()
        for acronym, words in acronym_words.items():
            exclude_acronym = False
            for word in words:
                # Verify if the word is present in the document
                word_found = False
                for token in doc:
                    if token.text.lower() == word:
                        word_found = True
                        # If the POS of the word is in the exclusion list, mark the acronym for exclusion
                        if token.pos_ in pos_to_exclude:
                            exclude_acronym = True
                            print(f"La palabra '{token.text}' en el acrónimo '{acronym}' tiene un POS excluido: {token.pos_}")
                            break
                # If the word is not found in the document, mark the acronym for exclusion
                if not word_found:
                    exclude_acronym = True
                    print(f"The code must not enter here!")
                    print(f"La palabra '{word}' del acrónimo '{acronym}' no se encuentra en el documento.")
                    break
            # If the acronym passes the POS filter, add it to the set of acronyms to keep
            if not exclude_acronym:
                acronyms_to_keep.add(acronym)
                print(f"Acrónimo '{acronym}' ha pasado el filtro POS.")

        print("\nAcrónimos a mantener después de aplicar el filtro POS:", acronyms_to_keep)

        # Filter acronyms based on NER
        acronyms_to_remove = set()
        for ent in doc.ents:
            ent_text_lower = ent.text.lower().strip()
            if ent.label_ in ner_labels_to_exclude:
                print(f"Entity '{ent.text}' detected with ner_labels_to_exclude: {ent.label_}")
                # Verificar si la entidad coincide con las palabras de los acrónimos a mantener
                for acronym, words in acronym_words.items():
                    if any(word == ent_text_lower for word in words):
                        acronyms_to_remove.add(acronym)
                        print(f"Acronym '{acronym}' has been deleted due to its NER: {ent.label_}")

        print("\nAcrónimos a eliminar después de aplicar el filtro NER:", acronyms_to_remove)

        # Filtered acronyms due to POS and NER criteria
        filtered_acronyms = list(acronyms_to_keep - acronyms_to_remove)
        return filtered_acronyms

'''
# Example usage
analyzer = NERTextAnalyzer()
text1 = "Servicios de arquitecto para la remodelación de instalaciones que con motivo del proyecto HOGEI URTE se realizarán por parte de Euskalduna Jauregia Palacio Euskalduna, S.A en Bilbao"
detected_acronyms = ["arquitecto", "instalaciones", "obras", "HOGEI URTE", "Bilbao"]

filtered_acronyms = analyzer.analyze_text(text1, detected_acronyms)
print(filtered_acronyms)
'''

