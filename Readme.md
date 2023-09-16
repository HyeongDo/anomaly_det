# 이상 탐지

```
이상 탐지 모델에 시계열로 이루어진 csv 파일로 학습하고, 다른 csv 파일을 입력하여 이상 탐지를 하는 프로젝트.

date 컬럼을 기준으로 학습과 추론을 진행.

FastAPI로 진행하였고 /docs에서 train, predict 직접 테스트 가능.

☁️ main.py에 FastAPI 통신 코드
☁️ utils 디렉토리에 학습 및 추론 도움 코드
☁️ model 디렉토리에 직접 모델링한 모델 클래스 

😄 Log 파일 읽은 후, 토근화 하여 CSV 파일로 변경.
😄 Train 후, model 생성.
😄 생성된 모델을 이용하여 Predict.
```

## 개발 환경
- windows11
- cpu ( no gpu )
- python
- FastAPI

## 환경 설정
```
pip install -r requirements.txt
```

## Start
```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
