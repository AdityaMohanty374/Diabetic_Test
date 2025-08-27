from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch
import torch.nn as nn
import joblib

app = FastAPI()

class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.relu(self.layer1(x))
        out = self.relu(self.layer2(out))
        out = self.layer3(out)
        return out

# Load model + scaler
n_features = 8
model = NeuralNet(n_features)
model.load_state_dict(torch.load("diabetes_model.pth", map_location=torch.device("cpu")))
model.eval()
sc = joblib.load("scaler.save")

# Request schema
class InputData(BaseModel):
    data: list

@app.post("/predict")
def predict(input: InputData):
    arr = np.array([input.data])
    arr_scaled = sc.transform(arr)
    tensor = torch.from_numpy(arr_scaled.astype(np.float32))
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits)
        cls = (prob >= 0.4).float()
    return {"probability": prob.item(), "class": int(cls.item())}
