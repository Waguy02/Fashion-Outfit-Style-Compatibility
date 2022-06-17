import logging
import os
import torch
import torchvision.models
from torch import nn
from constants import ROOT_DIR
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
class OutfitNetwork(nn.Module):
    def __init__(self, model_name="my_model",reset=False,load_best=True):
        super(OutfitNetwork, self).__init__()
        self.model_name = model_name
        self.reset = reset
        self.load_best = load_best
        self.setup_dirs()
        self.setup_network()
        if not self.reset:
            self.load_state()
    ##1. Defining network architecture
    def setup_network(self):
        """
        Initialize the network  architecture here
        #TODO : Next Attention head to enhance the embeddings contexts.
        @return:
        """
        self.input_size=512
        self.hidden_size=1540



        ##1. Defining the CNN
        backbone=torchvision.models.resnet18(pretrained=True)
        cnn=nn.Sequential(*list(backbone.children())[:-1])  #
        for layer in cnn.children():
            for param in layer.parameters():
                param.requires_grad = False
        self.cnn=nn.Sequential(cnn,nn.Flatten())

        ##2. Define the Lstm
        self.bilstm = nn.LSTM(bidirectional=True, input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1,batch_first=True)

        ##3. Defining CNN projector  from hidden to embedding
        self.projector=nn.Sequential(
            nn.Linear(1540,2048),
            nn.relu(),
            nn.Linear(2048,2048),
            nn.relu(),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.tanh()
        )

    ##2. Model Saving/Loading
    def load_state(self):
        """
        Load model
        :param self:
        :return:
        """

        print("Exist :", os.path.exists(self.save_best_file))
        if  self.load_best and os.path.exists (self.save_best_file):
            logging.info(f"Loading best model state : {self.save_file}")
            self.load_state_dict(torch.load(self.save_file,map_location=device))
            return 
            
        if os.path.exists(self.save_file):
            logging.info(f"Loading model state : {self.save_file}")
            self.load_state_dict(torch.load(self.save_file,map_location=device))
    def save_state(self,best=False):
        if best:
            torch.save(self.state_dict(), self.save_best_file)
        else:
            torch.save(self.state_dict(), self.save_file)

    ##3. Setupping directories for weights /logs ... etc
    def setup_dirs(self):
        """
        Checking and creating directories for weights storage
        @return:
        """
        self.save_path = os.path.join(ROOT_DIR, 'zoos')
        self.model_dir = os.path.join(self.save_path, self.model_name)
        self.save_file = os.path.join(self.model_dir, f"{self.model_name}.pt")
        self.save_best_file = os.path.join(self.model_dir, f"{self.model_name}_best.pt")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    #4. Forward call
    def forward(self, input):
        # In this function we pass the 3 images and get the 3 embeddings
        "Forward call here"
        ##Reshape for cnn single pass
        batch_size,sequence_length,channel,height,width=input.shape
        input=input.view(batch_size*sequence_length,channel,height,width)
        input_embeddings=torch.squeeze(self.cnn(input))


        input_embeddings=input_embeddings.view(batch_size,sequence_length,input_embeddings.shape[-1])


        hidden_states,_=self.bilstm(input_embeddings)


        hidden_states_left,hidden_states_right=torch.split(hidden_states, (self.hidden_size, self.hidden_size), dim=-1)

        #Projector pass to get target embeddings
        target_embeddings_left=self.projector(torch.reshape(hidden_states_left,(batch_size * sequence_length, self.hidden_size)))
        target_embeddings_right=self.projector(torch.reshape(hidden_states_right,(batch_size * sequence_length, self.hidden_size)))

        target_embeddings_left=target_embeddings_left.view(batch_size,sequence_length,target_embeddings_left.shape[-1])
        target_embeddings_right=target_embeddings_right.view(batch_size,sequence_length,target_embeddings_right.shape[-1])


        return input_embeddings,target_embeddings_left,target_embeddings_right







