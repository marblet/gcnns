from models import APPNP, GAT, GCN, GFNN, MaskedGCN, MixHop, PPNP, SGC
from data.data import load_data
from train import Trainer
from utils import preprocess_features

import random
import numpy as np
import torch

SEED = 18
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    data = load_data('cora')
    data.features = preprocess_features(data.features)
    model = GCN(data)
    trainer = Trainer(model, data, lr=0.01, weight_decay=5e-4, epochs=200, patience=10, niter=10, verbose=True)
    trainer.run()
