import argparse
import logging

import torch.nn
from torch.optim import Adam
from logger import setup_logger
from networks.outfit_network import OutfitNetwork
from training.trainer import Trainer
def cli():
    """
   Parsing args
   @return:
   """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--reset", "-r", action='store_true', default=False   , help="Start retraining the model from scratch")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.00001, help="Learning rate of Adam optimized")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--model_name", "-n",help="Name of the model. If not specified, it will be automatically generated")
    parser.add_argument("--num_workers", "-w", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--batch_size", "-bs", type=int, default=16, help="Batch size for training")
    parser.add_argument("--log_level", "-l", type=str, default="INFO")
    parser.add_argument("--autorun_tb","-tb",default=False,action='store_true',help="Autorun tensorboard")
    return parser.parse_args()


def main(args):
    model_name = "base_model" if args.model_name is None else args.model_name##AutoParsing model name
    network=OutfitNetwork()
    optimizer = Adam(network.parameters(), lr=args.learning_rate)
    loss=torch.nn.MSELoss()
    logging.info("Training : "+model_name)
    trainer=Trainer(network,loss,optimizer,args.epochs,args.batch_size,args.num_workers,args.epochs,args.autorun_tb)
    trainer.fit()
    
    

if __name__ == "__main__":
    args = cli()
    setup_logger(args)
    main(args)
 