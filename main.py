from models.gcn import create_gcn_model
from models.gat import create_gat_model
from models.sgc import create_sgc_model
from models.gfnn import create_gfnn_model
from models.masked_gcn import create_masked_gcn_model
from models.appnp import create_appnp_model
from models.ppnp import create_ppnp_model
from data.data import load_data
from train import run
from utils import preprocess_features

import random
import numpy as np
import torch

SEED = 17
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    data = load_data('cora')
    data.features = preprocess_features(data.features)
    model = create_gcn_model(data)
    run(data, model, lr=0.005, weight_decay=5e-4, epochs=200, patience=10, niter=10)
