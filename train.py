from model import AE
from data import get_training_dataloader,get_data_from_s3
import torch.nn.functional as F
import argparse
import torch
import torch.optim as optim 
import os
import logging
import sys


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def train(args, model, optimizer, train_loader):
    epochs = args.epochs
    lr = args.learning_rate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_cuda = args.use_cuda
    for epoch in range(epochs):
        logger.info(f'start ecoch:{epoch}')
        for idx, data in enumerate(train_loader, 0):
            imgs, _ = data
            imgs = imgs.to(device)
            out = model(imgs, )
            loss = F.binary_cross_entropy(out, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--use-cuda', type=bool, default=True)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    args, _ = parser.parse_known_args()
    logger.info(f'train data path:{args.train}')
    train_data = get_data_from_s3(bucket='yu-shengkuo-test', key_prefix= 'MNIST')
    train_loader = get_training_dataloader(training_data=train_data)
    net = AE()
    device = "cuda" if torch.cuda.is_available() and args.use_cuda else 'cpu'
    logger.info(f'train on {device}')
    train_loader = get_training_dataloader(args.train)
    optimizer=torch.optim.Adam(net.parameters(), lr = args.learning_rate)
    train(args, model= net, optimizer=optimizer, train_loader=train_loader)
    # ... train `model`, then save it to `model_dir`
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(net.state_dict(), f)



    