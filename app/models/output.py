
# from sqlmodel import SQLModel

# class ClassificationResponse(SQLModel):
#     sentence: str
#     result: str

class ClassificationResponse():
    def __init__(self, sentence:str, result:str):
        self.sentence = sentence
        self.result = result