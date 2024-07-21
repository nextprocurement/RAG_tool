import dspy

# @ TODO: Si usamos dspy para esto, lo suyo sería optimizar todos los componentes, esto es, hacer un teleprompter para cada módulo (AcronymDetector, AcronymExpander, etc.). Entonces, no habría que meter un ejemplo en la definición del módulo, ya que en teoría, el teleprompter ya se encarga de eso.
# @ TODO: Me genera duda lo siguiente: Ahora mismo las descripciones en la signature están en español. Está bien, dado que ahora mismo el dataset con el que trabajamos está en español. Pero el problema de esto es que no es escalable. Si en un futuro queremos trabajar con un dataset en inglés, tendríamos que cambiar todas las descripciones de las signatures. ¿No sería mejor poner las descripciones en inglés desde el principio? Así, si en un futuro queremos trabajar con un dataset en inglés, no tendríamos que cambiar nada. ¿Qué opinas? Eso sí, el fine-tuning va a ser un poco más complicado.
class AcronymDetector(dspy.Signature):
    """
    Detecta los acrónimos, abreviaturas y siglas que contenga el texto.
    """

    TEXTO = dspy.InputField(desc="Texto en español que puede contener o no, acrónimos, siglas y/o abreviaturas que deben detectarse")
    ACRONIMOS = dspy.OutputField(desc="Lista de acrónimos, siglas y/o abreviaturas. Un acrónimo, es un tipo de palabra formada a partir de la fusión de varias palabras.")

class AcronymDetectorModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.Predict(AcronymDetector)

    def forward(self, texto):
        response = self.generator(TEXTO=texto)
        return dspy.Prediction(ACRONIMOS=response.ACRONIMOS)