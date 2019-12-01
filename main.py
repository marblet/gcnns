from data.data import load_data
from models.gcn import create_gcn_model
from train import run
from utils import preprocess_features


if __name__ == '__main__':
    data = load_data('cora')
    data.features = preprocess_features(data.features)
    model, optimizer = create_gcn_model(data)
    run(data, model, optimizer)