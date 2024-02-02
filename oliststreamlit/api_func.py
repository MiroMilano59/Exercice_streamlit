from fastapi import FastAPI
import numpy as np 
import pandas as pd

app = FastAPI()

@app.get('/info')
def read_root():
    return {'message': 'Hello ! welcom on my dashborad!'}