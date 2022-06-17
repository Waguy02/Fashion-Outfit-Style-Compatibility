import json
import logging
import os
import shutil
import subprocess
import torch
from torch.utils.tensorboard import SummaryWriter
from constants import TENSORBOARD_DIR
from tqdm import tqdm
# CUDA for PyTorch
from my_utils import Averager
from networks.outfit_network import OutfitNetwork
from dataset.dataset import create_dataloader, DatasetType

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class Trainer:
    """
    Class to manage the full training pipeline
    """

    def __init__(self, network: OutfitNetwork, loss, optimizer, nb_epochs=10, batch_size=128, num_workers=4,
                 reset=False, autorun_tb=False):
        """
        @param network:
        @param dataset_name:
        @param images_dirs:
        @param loss:
        @param optimizer:
        @param nb_epochs:
        @param nb_workers: Number of worker for the dataloader
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.network = network
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.loss = loss
        self.nb_epochs = nb_epochs
        self.train_data_loader = create_dataloader(type=DatasetType.TRAIN, batch_size=batch_size,
                                                   num_workers=num_workers)
        self.valid_data_loader = create_dataloader(type=DatasetType.VALID, batch_size=batch_size, num_workers=num_workers)

        self.fitb_valid_data_loader=create_dataloader(type=DatasetType.FITB, batch_size=batch_size, num_workers=num_workers)


        self.tb_dir = os.path.join(TENSORBOARD_DIR, self.network.model_name)
        self.epoch_index_file = os.path.join(self.tb_dir, "epoch_index.json")

        if reset:
            if os.path.exists(self.tb_dir):
                shutil.rmtree(self.tb_dir)
        if not os.path.exists(self.tb_dir):
            os.makedirs(self.tb_dir)
        self.summary_writer = SummaryWriter(log_dir=self.tb_dir)
        self.start_epoch = 0
        if not reset and os.path.exists(self.epoch_index_file):
            with open(self.epoch_index_file, "r") as f:
                self.start_epoch = json.load(f)["epoch"] + 1
                self.nb_epochs += self.start_epoch
                logging.info("Resuming from epoch {}".format(self.start_epoch))
        self.autorun_tb = autorun_tb

    def run_tensorboard(self):
        """
        Launch tensorboard
        @return:
        """
        cmd = f"tensorboard --logdir '{self.tb_dir}' --host \"0.0.0.0\" --port 6007"
        _ = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None, shell=True)

    def fit(self):
        if self.autorun_tb: self.run_tensorboard()
        logging.info("Launch training on {}".format(device))
        self.network.to(device)
        itr = self.start_epoch * len(self.train_data_loader) * self.batch_size  ##Global counter for steps
        best_loss = 1e20  # infinity
        if os.path.exists(os.path.join(self.tb_dir, "best_model_info.json")):
            with open(os.path.join(self.tb_dir, "best_model_info.json"), "r") as f:
                best_model_info = json.load(f)
                best_loss = best_model_info["eval_loss"]

        ##Print the graph of the network
        s=next(iter(self.train_data_loader))
        # self.summary_writer.add_graph(self.network, s.to(device))

        for epoch in range(self.start_epoch, self.nb_epochs):  # Training loop
            self.network.train()

            """"
            0. Initialize loss and other metrics
            """
            running_loss = Averager()
            running_loss_right = Averager()
            running_loss_left = Averager()

            pbar=tqdm(self.train_data_loader, desc="Epoch {}".format(epoch))
            for _, batch in enumerate(pbar):
                """
                Training lopp
                """
                break
                itr += self.batch_size

                """
                1.Forward pass
                """
                input_embeddings, targets_embeddings_left, targets_embeddings_right = self.network(batch.to(device))

                """
                2.Loss computation and other metrics
                """
                loss_left = self.loss(input_embeddings[1:], targets_embeddings_left[:-1])
                loss_left_value = loss_left.cpu().item()
                running_loss_left.send(loss_left_value)

                loss_right = self.loss(input_embeddings[:-1], targets_embeddings_right[1:])
                loss_right_value = loss_right.cpu().item()
                running_loss_right.send(loss_right_value)

                loss = (loss_left + loss_right) / 2
                loss_value = loss.cpu().item()
                running_loss.send(loss_value)

                """
                3.Optimizing
                """
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                """
                4.Writing logs and tensorboard data, loss and other metrics
                """
                pbar.set_description("Epoch {}. Loss: {} , Loss left : {}, Loss right : {}".format(epoch,loss_value,loss_left_value,loss_right_value))
                self.summary_writer.add_scalar("Train/MSE Loss mean", loss_value, itr)
                self.summary_writer.add_scalar("Train/MSE Loss left", loss_left_value, itr)
                self.summary_writer.add_scalar("Train/MSE Loss right", loss_right_value, itr)

            epoch_loss = running_loss.value
            self.summary_writer.add_scalar("Train/epoch_loss", epoch_loss, epoch)

            epoch_loss_left = running_loss_left.value
            self.summary_writer.add_scalar("Train/epoch_loss_left", epoch_loss_left, epoch)

            epoch_loss_right = running_loss_right.value
            self.summary_writer.add_scalar("Train/epoch_loss_right", epoch_loss_right, epoch)

            # Saving the model at the end of the epoch is better than the preious best one
            self.network.save_state()

            epoch_loss_val,epoch_loss_left_val,epoch_loss_right_val,fitb_accuracy = self.eval(epoch)
            self.scheduler.step(epoch_loss_val)
            if epoch_loss_val < best_loss:
                logging.info("Saving the best model")
                best_loss = epoch_loss_val
                self.network.save_state(best=True)
                with open(os.path.join(self.tb_dir, "best_model_info.json"), "w") as f:
                    f.write(json.dumps({"train_loss": float(epoch_loss),
                                        "eval_loss": float(epoch_loss_val), "epoch": epoch
                                        }, indent=4))

            logging.info("Epoch {}. Loss: {} , Loss left : {}, Loss right : {},\ val_loss : {}, val_loss_left : {}, val_loss_right : {},fitb_accuracy :{}".format(epoch,epoch_loss,epoch_loss_left,epoch_loss_right,epoch_loss_val,epoch_loss_left_val,epoch_loss_right_val,fitb_accuracy))

    def eval(self, epoch):
        """
        Compute loss and metrics on a validation dataloader
        @return:
        """
        fitb_accuracy=self.eval_fitb(epoch)
        with torch.no_grad():
            self.network.eval()
            running_loss = Averager()
            running_loss_right = Averager()
            running_loss_left = Averager()

            for _, batch in enumerate(tqdm(self.valid_data_loader, desc=f"Eval  Epoch {epoch}/{self.nb_epochs}")):
                """
                Training lopp
                """
                """
                1.Forward pass
                """
                input_embeddings, targets_embeddings_left, targets_embeddings_right= self.network(batch.to(device))

                """
                2.Loss computation and other metrics
                """


                loss_left = self.loss(input_embeddings[1:], targets_embeddings_left[:-1])
                loss_left_value = loss_left.cpu().item()
                running_loss_left.send(loss_left_value)

                loss_right = self.loss(input_embeddings[:-1], targets_embeddings_right[1:])
                loss_right_value = loss_right.cpu().item()
                running_loss_right.send(loss_right_value)

                loss = (loss_left + loss_right) / 2
                loss_value = loss.cpu().item()
                running_loss.send(loss_value)


            epoch_loss = running_loss.value
            epoch_loss_left = running_loss_left.value
            epoch_loss_right = running_loss_right.value
            self.summary_writer.add_scalar("Val/epoch_loss", epoch_loss, epoch)
            self.summary_writer.add_scalar("Val/epoch_loss_left", epoch_loss_left, epoch)
            self.summary_writer.add_scalar("Val/epoch_loss_right", epoch_loss_right, epoch)

            return epoch_loss,epoch_loss_left,epoch_loss_right,fitb_accuracy



    def eval_fitb(self, epoch):
        """
        Run fill in the blank evaluatoin

        """
        with torch.no_grad():
            self.network.eval()
            fitb_accuracy=Averager()
            desc=f"Eval FITB {epoch + 1}/{self.nb_epochs}"
            pbar=tqdm(self.fitb_valid_data_loader, desc=desc)
            for _batch, batch in enumerate(pbar):
                left=batch["left"]
                right=batch["right"]
                proposals=batch["proposals"]
                _,_,pred_left=self.network(left.to(device))
                _,pred_right,_=self.network(right.to(device))

                # pred_left=torch.stack([pred_left[i][-1] for i in range(len(mask_idx))])
                # pred_right=torch.stack([pred_right[i][0] for i in range(len(mask_idx))])
                pred_left=pred_left[:,-1,:]
                pred_right=pred_right[:,0,:]

                pred_embedding=1/2*(pred_left+pred_right) ## Stacking leftward and backward prediction at opsitiv
                pred_embedding=pred_embedding.squeeze(0)



                proposals_input=proposals.view(proposals.shape[0]*proposals.shape[1],3,224,224)
                proposals_embeddings=self.network.cnn(proposals_input.to(device))
                proposals_embeddings=proposals_embeddings.view(proposals.shape[0],proposals.shape[1],-1)
                pred_embedding=pred_embedding.unsqueeze(1).repeat(1,proposals_embeddings.shape[1],1)
                distance_to_proposals=(torch.norm(pred_embedding-proposals_embeddings,dim=2)**2)/proposals_embeddings.shape[2]
                closest_idx=torch.argmin(distance_to_proposals,dim=1)
                correct_count=torch.sum(closest_idx==distance_to_proposals.shape[1]-1)##Check if the closes is the masked item
                acc=(correct_count/distance_to_proposals.shape[0])*100
                fitb_accuracy.send(acc)
                pbar.set_description(f"Val Epoch {epoch + 1}/{self.nb_epochs} - FITB  Acc: {acc:.4f}")
            fitb_accuracy_value=fitb_accuracy.value
            self.summary_writer.add_scalar("Val/fitb_accuracy", fitb_accuracy_value, epoch)
            return fitb_accuracy_value
