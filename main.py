from fastapi import FastAPI, UploadFile
from utils.utils import load_and_preprocess_data, train_model, initialize_model, inference
from utils.utils import scaler

app = FastAPI()

@app.post("/train")
async def train(file: UploadFile):
    '''
    :param file: csv 파일
    :return:
        학습과 추론 파일은 같은 shape을 가지고 진행합니다.
        기준 컬럼은 date로 합니다.
    '''
    try:
        with open('training_data.csv', 'wb') as f:
            f.write(file.file.read())

        training_data = load_and_preprocess_data('training_data.csv')
        train_model(training_data)
    except:
        return {"message": 'file read fail.'}
    
    return {"message": "train Success."}


@app.post("/predict")
async def predict(file: UploadFile, threshold: float):
    '''
    :param file: csv 파일
    :param threshold: 이상 탐지 임계치 설정 [0,1]
    :return:
        학습과 추론은 같은 shape을 가지고 진행합니다.
        기준 컬럼은 date로 합니다.
    '''
    try:
        with open('inference_data.csv', 'wb') as f:
            f.write(file.file.read())

        inference_data = load_and_preprocess_data('inference_data.csv')
        input_dim = inference_data.shape[1]

    except:
        return {"message": "file read fail."}

    try:
        model = initialize_model(input_dim)
    except:
        return {"message": "model initialize fail."}

    try:
        X_inference = scaler.transform(inference_data)
        anomalies, mse_scores = inference(model, X_inference, threshold)
    except:
        return {"message": 'predict fail.'}

    return {"anomalies": anomalies.tolist(), "mse_scores": mse_scores.tolist()}
