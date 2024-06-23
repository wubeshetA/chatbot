## POST API for to response the user  
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chat import main_
# import nltk

app = FastAPI()
# port = 8080
# Allow all origins
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class QA(BaseModel):
    question: str


@app.post('/api/predict')
def predict(request: QA):
    """
        requestBody contain the question
        in a string fromat and it return the response
    """
    msg = main_(request.question.lower())
    print(msg)
    return {'data': msg}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)


