from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from utils.utils import load_and_preprocess_data, train_model, initialize_model, inference, tokenize_log_line
from utils.utils import scaler
from io import StringIO

import pandas as pd

app = FastAPI()


@app.post("/read_log")
async def read_and_tokenize_log(file: UploadFile):
    '''
    로그 파일 읽고 csv로 변경.
    :param file: 텍스트 파일
    :return: csv 파일
    '''
    try:
        log_text = file.file.read().decode("utf-8")

        log_lines = log_text.strip().split('\n')
        log_tokens = [tokenize_log_line(line) for line in log_lines]

        log_df = pd.DataFrame(log_tokens, columns=['date'] + [f'col{i}' for i in range(1, len(log_tokens[0]))])

        temp_csv_file = 'temp_log_data.csv'
        log_df.to_csv(temp_csv_file, index=False)

        return FileResponse(temp_csv_file, media_type='text/csv',
                            headers={"Content-Disposition": "attachment; filename=results_split_log.csv"})

    except Exception as e:
        return {"message": f"log read fail: {str(e)}"}


@app.post("/train")
async def train(file: UploadFile):
    '''
        학습과 추론 파일은 같은 shape을 가지고 진행합니다.
        기준 컬럼은 date로 합니다.
    :param file: csv 파일
    :return: succes, fail
    '''
    try:
        with open('training_data.csv', 'wb') as f:
            f.write(file.file.read())

        training_data = load_and_preprocess_data('training_data.csv')
        train_model(training_data)
    except Exception as e:
        return {"message": f'file read fail. {str(e)}'}
    
    return {"message": "train Success."}


@app.post("/predict")
async def predict(file: UploadFile, threshold: float):
    '''
        학습과 추론은 같은 shape을 가지고 진행합니다.
        기준 컬럼은 date로 합니다.
    :param file: csv 파일
    :param threshold: 이상 탐지 임계치 설정 [0,1]
    :return: success {anomalies list, mse_scores list}
    '''
    try:
        with open('inference_data.csv', 'wb') as f:
            f.write(file.file.read())

        inference_data = load_and_preprocess_data('inference_data.csv')
        input_dim = inference_data.shape[1]

    except Exception as e:
        return {"message": f"file read fail. {str(e)}"}

    try:
        model = initialize_model(input_dim)
    except Exception as e:
        return {"message": f"model initialize fail. {str(e)}"}

    try:
        X_inference = scaler.transform(inference_data)
        anomalies, mse_scores = inference(model, X_inference, threshold)
    except Exception as e:
        return {"message": f'predict fail.{str(e)}'}

    return {"anomalies": anomalies.tolist(), "mse_scores": mse_scores.tolist()}
