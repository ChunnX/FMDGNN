import torch

N_SUBJECTS = 678
N_ROI = 35
# N_FEATURE = 595
N_FEATURE = int(N_ROI * (N_ROI - 1) / 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_path = "saved_trained/"

# parameters
fed_rounds = 10
epochs_per_round = 10
batch_size = 32
c_idx = 0
kl = False
topo = False
lamb = [0.1, 0.1]