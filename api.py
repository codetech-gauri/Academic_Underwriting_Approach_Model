
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load('loan_approval_model.pkl') 

class Applicant(BaseModel):
    FPS: float

@app.post('/predict')
def predict(applicant: Applicant):
    prediction = model.predict([[applicant.FPS]])
    return {'approval': int(prediction[0])}
