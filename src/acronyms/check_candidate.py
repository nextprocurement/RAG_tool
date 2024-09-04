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
        Analyze the text and provide grammatical and NER details of detected acronyms.

        Parameters:
        - text (str): The text to analyze.
        - detected_acronyms (list): A list of acronyms detected in the text.

        Returns:
        - Dictionary containing details of words and entities that match detected acronyms.
        """
        # Detect language of the text and load the corresponding spaCy model
        language_detector = LanguageDetectorModule()
        detected_language = language_detector.detect_language(text)
        print("Detected language is:", detected_language)
        # Load the appropriate model based on the detected language
        self.load_model(detected_language)

        # Process the text with the spaCy model
        doc = self.nlp(text)
        # Convert detected acronyms to lowercase for consistent comparisons
        detected_acronyms = [acronym.lower() for acronym in detected_acronyms]

        # Filter tokens to exclude certain POS types from the list of acronyms
        filtered_acronyms = detected_acronyms[:]  # Create a copy of the detected acronyms
        for token in doc:
            if token.text.lower() in filtered_acronyms:
                # Print the type of POS for each detected acronym
                print(f"Acronym '{token.text}' detected with POS: {token.pos_}")
                if token.pos_ in ["ADJ", "PUNCT", "VERB"]:
                    filtered_acronyms.remove(token.text.lower())
                    print(f"Acronym '{token.text}' has been deleted from the list due to POS: {token.pos_}")

        # Apply NER and filter entities based on the criteria
        for ent in doc.ents:
            if ent.text.lower() in filtered_acronyms:
                # Print the NER label of each detected acronym
                print(f"Acronym '{ent.text}' detected with NER label: {ent.label_}")
                # If it's LOC or GPE but has 2 or fewer letters, do not remove it
                if ent.label_ in ["LOC", "GPE"] and len(ent.text) <= 2:
                    print(f"Acronym '{ent.text}' is of type {ent.label_} with 2 or fewer letters and will not be removed.")
                    continue
                if ent.label_ in ["PER", "LOC", "GPE", "TIME", "MONEY"]:
                    filtered_acronyms.remove(ent.text.lower())
                    print(f"Acronym '{ent.text}' has been deleted from the list due to NER entity: {ent.label_}")
        
        # Initialize lists to collect word and entity information
        word_details = []
        entities_details = []

        # Collect POS and tag details for filtered acronyms
        for token in doc:
            if token.text.lower() in filtered_acronyms:
                word_info = {
                    "text": token.text,
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "dep": token.dep_,
                }
                word_details.append(word_info)

        # Collect entity information for remaining acronyms
        for ent in doc.ents:
            if ent.text.lower() in filtered_acronyms:
                entity_info = {
                    "text": ent.text,
                    "label": ent.label_,
                }
                entities_details.append(entity_info)

        # Return the collected information
        return {"words": word_details, "entities": entities_details}


'''
# Example usage
analyzer = NERTextAnalyzer()
text1 = "Presp. de ejecución de obras para impermeabilización de cubierta plana de patio interior del edif. anexo a la Comisaría Local de Santiago de Compostela. Coruña"
detected_acronyms = ["VP", "PK", "LP", "A-6", "suminitro", "FITUR", "Renedo de Esgueva", "Villabañez", "Coruña"]

word_details = analyzer.analyze_text(text1, detected_acronyms)

print("Resultados del Análisis del Texto 1:")
print("\nPalabras Analizadas:")
for word_info in word_details['words']:
    print(f"Texto: {word_info['text']}, POS: {word_info['pos']}, Tag: {word_info['tag']}, Dep: {word_info['dep']}")

print("\nEntidades Detectadas:")
for entity_info in word_details['entities']:
    print(f"Texto: {entity_info['text']}, Etiqueta: {entity_info['label']}")
'''
