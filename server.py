# src/server.py
from fastapi import FastAPI
from pydantic import BaseModel
from infer import pipeline

app = FastAPI(title="Gloss-QA Pipeline")

class Req(BaseModel):
    gloss: str

@app.post("/predict")
def predict(req: Req):
    out_gloss = pipeline(req.gloss)
    return {"gloss": out_gloss}


