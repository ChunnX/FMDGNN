from config import *
from utils import *
from model import *
from cross_val import *


file_dir = "data LH/data LH"
sizes = [192, 192, 192, 102]
idx_list, hospital_domains_list, dataset_list = get_dataset(file_dir, sizes=sizes, seed=42)
global_test = dataset_list[-1]
cross_datasets = get_cross_datasets(idx_list, n_folds=4)

if not res:
    encoder = GCNencoder(nfeat=595, nhid=32)
    decoders = [GCNdecoder(nhid=32, nfeat=595) for _ in range(5)]
else:
    encoder = ResGCNencoder(nfeat=595, nhid=32)
    decoders = [ResGCNdecoder(nhid=32, nfeat=595) for _ in range(5)]

hospital_1 = Hospital(encoder=encoder, decoders=decoders, views=[1, 3]).to(device)
hospital_2 = Hospital(encoder=encoder, decoders=decoders, views=[1, 2, 5]).to(device)
hospital_3 = Hospital(encoder=encoder, decoders=decoders, views=[2, 3, 4, 5]).to(device)

hospital_list = [hospital_1, hospital_2, hospital_3]

# train the model
# cross_results = cross_validation(hospital_list=hospital_list,
#                                  cross_datasets=cross_datasets,
#                                  global_test=global_test,
#                                  n_folds=n_folds,
#                                  res=res, aggr=aggr)

# save_cross_results(cross_results)

# Using trained model:
model_name = "GCN_FMTL_FedAvg"
cross_results = load_cross_results(filename=f"{save_path}{model_name}")
print_fed_results(cross_results[-2], cross_results[2])

