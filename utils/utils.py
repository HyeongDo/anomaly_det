from model.AnomalyDetectionModel import AnomalyDetectionModel
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

import pandas as pd
import torch.nn as nn
import torch

batch_size = 64
threshold = 0.1

scaler = StandardScaler()
def train_model(training_data):
    input_dim = training_data.shape[1]
    model = AnomalyDetectionModel(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    epochs = 100

    X_train = scaler.fit_transform(training_data)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 모델 학습
    for epoch in range(epochs):
        for data in train_loader:
            inputs = data[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), 'anomaly_detection_model.pth')


def initialize_model(input_dim):
    model = AnomalyDetectionModel(input_dim)
    try:
        model.load_state_dict(torch.load('anomaly_detection_model.pth'))
        model.eval()
        return model
    except FileNotFoundError:
        return None


def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['date_column'] = (pd.to_datetime(data['date']) - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    return data.drop(columns=['date']).values


def inference(model, data, threshold):
    with torch.no_grad():
        recon_data = model(torch.tensor(data, dtype=torch.float32))
        mse_loss = torch.mean((torch.tensor(data, dtype=torch.float32) - recon_data) ** 2, dim=1).numpy()
        anomalies = mse_loss > threshold
    return anomalies, mse_loss
