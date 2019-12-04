from data.data import load_data
from models.gcn import create_gcn_model
from models.gat import create_gat_model
from models.sgc import create_sgc_model
from models.gfnn import create_gfnn_model
from models.graphsage import create_graphsage_model
from models.masked_gcn import create_masked_gcn_model
from train import run
from utils import preprocess_features


if __name__ == '__main__':
    data = load_data('cora')
    data.features = preprocess_features(data.features)
    model, optimizer = create_gcn_model(data)
    run(data, model, optimizer, verbose=True)
