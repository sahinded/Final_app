from fastapi import FastAPI, Request
import pickle
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel, conlist
from typing import List
from fastapi import FastAPI, Request, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np

import uvicorn 
import pandas as pd

templates = Jinja2Templates(directory='templates')
app=FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")



class Iris(BaseModel):
    data: List[conlist(float, min_items=4, max_items=4)]



@app.get("/", response_class=HTMLResponse)
async def home_page(request: Request):
    return templates.TemplateResponse("index.html", {"request":request})

@app.post("/predict")
async def make_prediction(request: Request, SepalLength:str = Form(), SepalWidth:str = Form(), PetalLength:str = Form(), PetalWidth:str=Form()):
    testData = np.array([SepalLength, SepalWidth, PetalLength, PetalWidth]).reshape(-1, 4)
    model = joblib.load(open('model/final_model_joblib','rb'))
    predictions = model.predict(testData).tolist()
    return predictions

    

if __name__=="__main__":
    uvicorn.run("prediction:app",reload=True)