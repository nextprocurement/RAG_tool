import dspy 

class AcronymValidator(dspy.Signature):
    """
    Verifica si la palabra es un acrónimo, sigla o abreviatura 'true', o 'false' por el contrario. 
    """
    ACRONIMO = dspy.InputField(desc = "Acrónimo, sigla o abreviatura pendiente de verificar")
    CONTEXTO = dspy.InputField()
    RESULTADO = dspy.OutputField()

class AcronymValidatorModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generator = dspy.Predict(AcronymValidator)

    def forward(self, acronimo, contexto):
        # Llamar al modelo con cada acrónimo y contexto asociado
        response = self.generator(ACRONIMO=acronimo, CONTEXTO=contexto)
        # Imprimir la respuesta para verificar su estructura
        print("Response:", response.RESULTADO)
        
        return dspy.Prediction(RESULTADO = response.RESULTADO)