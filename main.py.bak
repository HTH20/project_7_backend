from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn

app = FastAPI()

class DataScoring(BaseModel):
    CODE_GENDER_F : int
    CODE_GENDER_M : int
    CODE_REJECT_REASON_CLIENT_mean : float
    CODE_REJECT_REASON_HC_mean : float
    CODE_REJECT_REASON_LIMIT_mean : float
    CODE_REJECT_REASON_SCOFR_mean : float
    CODE_REJECT_REASON_SCO_mean : float
    CODE_REJECT_REASON_SYSTEM_mean : float
    CODE_REJECT_REASON_VERIF_mean : float
    CODE_REJECT_REASON_XAP_mean : float
    CODE_REJECT_REASON_XNA_mean : float
    EXT_SOURCE_2 : float
    EXT_SOURCE_3 : float
    FLAG_OWN_CAR_N : int
    FLAG_OWN_CAR_Y : int
    NAME_EDUCATION_TYPE_Academic_degree : int
    NAME_EDUCATION_TYPE_Higher_education : int
    NAME_EDUCATION_TYPE_Incomplete_higher : int
    NAME_EDUCATION_TYPE_Lower_secondary : int
    NAME_EDUCATION_TYPE_Secondary_secondary_special : int

@app.post('/predict')
async def predict(data_scoring: DataScoring):
    data = data_scoring.dict()
    pickle_in = open("test_model.pkl", "rb")
    model = pickle.load(pickle_in)
    data_in = [[data['CODE_GENDER_F'], data['CODE_GENDER_M'], data['CODE_REJECT_REASON_CLIENT_mean'], data['CODE_REJECT_REASON_HC_mean'], data['CODE_REJECT_REASON_LIMIT_mean'], data['CODE_REJECT_REASON_SCOFR_mean'], data['CODE_REJECT_REASON_SCO_mean'], data['CODE_REJECT_REASON_SYSTEM_mean'], data['CODE_REJECT_REASON_VERIF_mean'], data['CODE_REJECT_REASON_XAP_mean'], data['CODE_REJECT_REASON_XNA_mean'], data['EXT_SOURCE_2'],data['EXT_SOURCE_3'], data['FLAG_OWN_CAR_N'], data['FLAG_OWN_CAR_Y'], data['NAME_EDUCATION_TYPE_Academic_degree'], data['NAME_EDUCATION_TYPE_Higher_education'], data['NAME_EDUCATION_TYPE_Incomplete_higher'], data['NAME_EDUCATION_TYPE_Lower_secondary'], data['NAME_EDUCATION_TYPE_Secondary_secondary_special']]]
   
    prediction = model.predict(data_in)
    probability = model.predict_proba(data_in).max()
    return  prediction.tolist()
    
if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)