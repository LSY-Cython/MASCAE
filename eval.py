from torch.utils.data import DataLoader
from model.scae import MASCAE
from test import *
from dataloader import *

model_name = "MASCAE"
model = MASCAE(n_channels=38, init_nc=8, mem_dim=25)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(3)
dataset = "smd"
train_loader = data_loader("datafile/train_smd_data.txt")
normal_loader = data_loader("datafile/normal_smd_data.txt")
anomaly_loader = data_loader("datafile/anomaly_smd_data.txt")

for i in range(0, 30, 1):
    weight_path = f"weights/{model_name}/{dataset}/epoch{i*5}.pt"
    weight = torch.load(weight_path, map_location=device)
    model.load_state_dict(weight)
    model.to(device)
    model.eval()
    print(f"------epoch{i*5}------")
    testing(model=model,
            train_loader=train_loader,
            normal_loader=normal_loader,
            anomaly_loader=anomaly_loader,
            device=device,
            dataset=dataset)