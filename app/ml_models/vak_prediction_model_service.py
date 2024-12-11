from app.ml_models.model_training_service import VAKClassificationModelTraining
from app.models.output import ClassificationResponse


class VAKClassificationModelService:
    def __init__(self):
        self.service = VAKClassificationModelTraining()

    async def classify(self, sentence)->ClassificationResponse:
        if self.service.model is None:
            return 'Model not found'
        result = await self.service.classify(sentence)
        return ClassificationResponse(sentence=sentence, result=result)

if __name__ == '__main__':
    service = VAKClassificationModelService()
    print(service.classify('I like watching movies'))
