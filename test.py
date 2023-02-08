import torch
from utils import *

def testing(model,train_loader,normal_loader,anomaly_loader,device,dataset):
    with torch.no_grad():
        error_normal_set,error_anomaly_set,error_train_set = [],[],[]
        query_normal_set,query_anomaly_set,query_train_set = [],[],[]
        update_normal_set,update_anomaly_set,update_train_set = [],[],[]
        rec_normal_set,rec_anomaly_set,rec_train_set = [],[],[]
        for mvts_normal in normal_loader:
            mvts_normal = mvts_normal.to(device)
            rec_loss_normal,fea_loss_normal,sep_loss_normal,enc_loss_normal,query_normal,update_normal,rec_normal = model(mvts_normal)
            error_normal = rec_loss_normal.item()
            error_normal_set.append(error_normal)
            query_normal = np.array(torch.flatten(query_normal).cpu().squeeze()).tolist()
            query_normal_set.append(query_normal)
            update_normal = np.array(torch.flatten(update_normal).cpu().squeeze()).tolist()
            update_normal_set.append(update_normal)
            rec_normal = np.array(rec_normal.cpu())
            rec_normal_set.append(rec_normal)
        for mvts_anomaly in anomaly_loader:
            mvts_anomaly = mvts_anomaly.to(device)
            rec_loss_anomaly,fea_loss_anomaly,sep_loss_anomaly,enc_loss_anomaly,query_anomaly,update_anomaly,rec_anomaly = model(mvts_anomaly)
            error_anomaly = rec_loss_anomaly.item()
            error_anomaly_set.append(error_anomaly)
            query_anomaly = np.array(torch.flatten(query_anomaly).cpu().squeeze()).tolist()
            query_anomaly_set.append(query_anomaly)
            update_anomaly = np.array(torch.flatten(update_anomaly).cpu().squeeze()).tolist()
            update_anomaly_set.append(update_anomaly)
            rec_anomaly = np.array(rec_anomaly.cpu())
            rec_anomaly_set.append(rec_anomaly)
        for mvts_train in train_loader:
            mvts_train = mvts_train.to(device)
            rec_loss_train,fea_loss_train,sep_loss_train,enc_loss_train,query_train,update_train,rec_train = model(mvts_train)
            error_train = rec_loss_train.item()
            error_train_set.append(error_train)
            query_train = np.array(torch.flatten(query_train).cpu().squeeze()).tolist()
            query_train_set.append(query_train)
            update_train = np.array(torch.flatten(update_train).cpu().squeeze()).tolist()
            update_train_set.append(update_train)
            rec_train = np.array(rec_train.cpu())
            rec_train_set.append(rec_train)
    error_set = {"train": error_train_set, "normal": error_normal_set, "anomaly": error_anomaly_set}
    with open(f"{dataset}_reconstruction.json", "w") as f:
        json.dump(error_set, f)
    evaluation(f"{dataset}_reconstruction.json")