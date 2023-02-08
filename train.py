from torch import nn
from model.scae import MASCAE
import matplotlib.pyplot as plt
import time
from dataloader import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
epoch = 150
init_lr = 1e-3
alpha = 0.01
beta = 0.01
gama = 0.005
dataset = "smd"
data_file = "datafile/train_smd_data.txt"
train_loader = data_loader(data_file)
model_name = "MASCAE"
recon_loss_func = nn.MSELoss().to(device)
model = MASCAE(n_channels=38, init_nc=16, mem_dim=25)
model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

# Train MASCAE
epoch_rec_loss = list()
epoch_fea_loss = list()
epoch_sep_loss = list()
epoch_enc_loss = list()
epoch_total_loss = list()
for epoch_idx in range(epoch):
    batch_rec_loss = list()
    batch_fea_loss = list()
    batch_sep_loss = list()
    batch_enc_loss = list()
    batch_total_loss = list()
    start_time = time.time()
    for mvts in train_loader:
        mvts = mvts.to(device)
        rec_loss, fea_loss, sep_loss, enc_loss, query, update = model(mvts)
        total_loss = rec_loss + alpha*fea_loss + beta*sep_loss + gama*enc_loss
        batch_rec_loss.append(rec_loss.item())
        batch_fea_loss.append(fea_loss.item())
        batch_sep_loss.append(sep_loss.item())
        batch_enc_loss.append(enc_loss.item())
        batch_total_loss.append(total_loss.item())
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()
    end_time = time.time()
    epoch_rec_loss.append(np.mean(batch_rec_loss))
    epoch_fea_loss.append(np.mean(batch_fea_loss))
    epoch_sep_loss.append(np.mean(batch_sep_loss))
    epoch_enc_loss.append(np.mean(batch_enc_loss))
    epoch_total_loss.append(np.mean(batch_total_loss))
    print(f"[epoch {epoch_idx}/{epoch}] " + f"rec_loss={epoch_rec_loss[-1]:.5f} " +
          f"fea_loss={epoch_fea_loss[-1]:.5f} " + f"sep_loss={epoch_sep_loss[-1]:.5f} " +
          f"enc_loss={epoch_enc_loss[-1]:.5f} " + f"total_loss={epoch_total_loss[-1]:.5f} " +
          f"time={end_time-start_time:.5f}s")
    if (epoch_idx+1) % 5 == 0:
        torch.save(model.state_dict(), f"weights/{model_name}/{dataset}/epoch{epoch_idx+1}.pt")
plt.subplot(5, 1, 1)
plt.plot(epoch_rec_loss, color="blue", label="rec_loss")
plt.legend()
plt.subplot(5, 1, 2)
plt.plot(epoch_fea_loss, color="green", label="feature_loss")
plt.legend()
plt.subplot(5, 1, 3)
plt.plot(epoch_sep_loss, color="purple", label="separation_loss")
plt.legend()
plt.subplot(5, 1, 4)
plt.plot(epoch_enc_loss, color="yellow", label="enc_loss")
plt.legend()
plt.subplot(5, 1, 5)
plt.plot(epoch_total_loss, color="red", label="total_loss")
plt.legend()
plt.savefig(f"{model_name}_train_{dataset}_loss.png")