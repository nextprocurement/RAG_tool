import spacy

class SpacyTextAnalyzer:
    def __init__(self, model="es_core_news_md"):
        self.nlp = spacy.load(model)

    def analyze_text(self, text, detected_acronyms):
        '''
        Analyze the text and provide grammatical details of each word that are in the detected_acronyms list.
        
        Parameters:
        - text (str): The text to analyze.
        - detected_acronyms (list): A list of acronyms detected in the text.
        '''
        doc = self.nlp(text)
        
        # Convert detected_acronyms to lowercase for consistent comparison
        detected_acronyms = [acronym.lower() for acronym in detected_acronyms]
        
        word_details = []
        for token in doc:
            # Check if the token text matches any of the detected acronyms
            if token.text.lower() in detected_acronyms:
                word_info = {
                    "text": token.text,
                    "lemma": token.lemma_,
                    "pos": token.pos_,      
                    "tag": token.tag_,     
                    "dep": token.dep_,      
                }
                word_details.append(word_info)

        return word_details

analyzer = SpacyTextAnalyzer()
text1 = "Refuerzo de Firme en la VP 3001 Renedo de Esgueva a Pesquera de Duero. Tramo: Villabañez a PK 12+422. El suminitro de mezclas bituminosas se realizará en la planta de la empresa suministradora"
text2 = "Refuerzo de Firme en la VP 6603 Mota del Marqués a LP Zamora por Casasola se Arión. Tramo: A-6 a Villalbarba"
text3 = "Refuerzo de firme en la VP 4013 Melgar de Arriba a Villacarralón. Tramo: Santervás a Villacarralón"
text4 = "Desarrollo del programa de intervención socioeducativa y psicosocial en Eibar"
text5 = "STAND DE EUSKADI EN FITUR Y SUS POSIBLES ADAPTACIONES AL RESTO DE FERIAS NACIONALES E INTERNACIONALES"

detected_acronyms = ["VP", "PK", "LP", "A-6", "suminitro", "FITUR", "Eibar", "RESTO", "NACIONALES"]
# Ejecutar el análisis gramatical del texto
word_details = analyzer.analyze_text(text1, detected_acronyms)
print("Detalles de las palabras en el texto:")
for detail in word_details:
    print(detail)