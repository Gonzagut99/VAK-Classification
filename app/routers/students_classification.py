from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.ml_models.vak_prediction_model_service import VAKClassificationModelService
from app.models.input import InputSentence
from app.routers.ReponseModel import ResponseModel

classify_student_router = APIRouter(prefix='/dl')

service = VAKClassificationModelService()
@classify_student_router.post("/predictvak", status_code=status.HTTP_201_CREATED, tags=["VAKClassification"])    
async def predict(body:InputSentence):
    try:
        response = await service.classify(body.sentence)
        print(response)
        return JSONResponse(
            content=ResponseModel(
                message='Prediction successful',
                code=201,
                data={
                    'sentence': response.sentence,
                    'result': response.result
                }
            ).get_serialized_response()
        )
    except Exception as e:
        return JSONResponse(
            content=ResponseModel(
                message='Prediction failed',
                code=500,
                error=True,
                detail=str(e)
            ).get_serialized_response()
        )
