import torch
from sklearn.metrics import mean_absolute_error

from losses import *
from config import *


def evaluate(hospital, test_loader, device):
    print("Start Testing ...")
    test_loss = 0
    view_losses = [0, 0, 0, 0, 0]
    BCs, ECs, PCs = 0, 0, 0

    hospital.eval()
    with torch.no_grad():
        for src, tgt1, tgt2, tgt3, tgt4, tgt5 in test_loader:
            torch.cuda.empty_cache()
            tgts = [tgt1, tgt2, tgt3, tgt4, tgt5]
            hospital_tgts = [tgts[int(view) - 1] for view in hospital.decoders.keys()]
            hospital_views = [int(view) for view in hospital.decoders.keys()]
            edge_idx = torch.Tensor([[i for i in range(src.shape[0])]] * 2).long().to(device)

            hospital.eval()
            outputs = hospital(src, edge_idx)
            loss_list = [reconstruction_loss(outputs[str(view)], tgt) for view, tgt in
                         zip(hospital_views, hospital_tgts)]

            # compute centrality
            BC, EC, PC = 0, 0, 0
            for view, tgt in zip(hospital_views, hospital_tgts):
                pred = outputs[str(view)]
                topo_pred = topological_measures(pred)
                topo_tgt = topological_measures(tgt)
                BC += mean_absolute_error(topo_pred[0], topo_tgt[0])
                EC += mean_absolute_error(topo_pred[1], topo_tgt[1])
                PC += mean_absolute_error(topo_pred[2], topo_tgt[2])

            BCs += BC
            ECs += EC
            PCs += PC

            for view, l in zip(hospital_views, loss_list):
                view_losses[view - 1] += l.detach().cpu().numpy()
            loss = sum(loss_list)
            test_loss += loss.item()

        test_loss /= len(test_loader)
        # CCs /= len(test_loader)
        BCs /= len(test_loader)
        ECs /= len(test_loader)
        PCs /= len(test_loader)

        for i in range(len(view_losses)):
            view_losses[i] /= len(test_loader)

        print(f"Testing Loss: {test_loss}")
        topology = [BCs, ECs, PCs]

    return test_loss, view_losses, topology


