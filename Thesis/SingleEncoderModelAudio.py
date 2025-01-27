import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

class SingleEncoderModelAudio(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_dim, 
                 num_layers, 
                 dropout_rate, 
                 output_size: int):
        super(SingleEncoderModelAudio, self).__init__()
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate    
        self.output_size = int(output_size)
        # self.FUSION_TEHNIQUE = FUSION_TEHNIQUE
        # print("input_size:", input_size)
        
        # GRU layer
        self.gru = nn.GRU(input_size=self.input_size, 
                          hidden_size=self.hidden_dim,  # Correct parameter is hidden_size
                          num_layers=self.num_layers, 
                          batch_first=True,  # Input is [batch_size, seq_length, input_size]
                          dropout=self.dropout_rate if self.num_layers > 1 else 0)
        # print(f"""\n----INSIDE AUDIO--- 
        #       input_size= {input_size}, 
        #       hidden_dim= {hidden_dim}, 
        #       num_layers= {num_layers}, 
        #       dropout_rate= {dropout_rate}, 
        #       output_size = {output_size}\n""")

        # Fully connected output layer
        # print(f"Initializing nn.Linear with hidden_dim={self.hidden_dim}, output_size={self.output_size}")

        self.fc2 = nn.Linear(self.hidden_dim, self.output_size)
        
        # Tanh activation function to scale outputs between -3 and 3
        self.tanh = nn.Tanh()

    def forward(self, x, lengths):
        batch_size = lengths.size(0)
        x = x.float()
        # print("x shape audio:", x.shape) # x` shape: x shape audio: torch.Size([56, 32, 74]) (batch_size, seq_len, input_size)


        # GRU forward pass
        output, hidden = self.gru(x)  # `hidden` shape: (num_layers, batch_size, hidden_dim)
        # print("1hidden shape audio:", hidden.shape)
        # print("output shape audio:", output.shape)

        last_hidden = hidden[-1]  # Use the last hidden state
        # print("2hidden shape audio:", last_hidden.shape)
        # Fully connected layers
        output1 = self.fc2(last_hidden)  # Output of shape [batch_size, output_size]
        # print("output1 shape audio:", output1.shape)
        # Apply tanh and scale the output to [-3, 3]
        output2 = 3 * self.tanh(output1)
        # print("output2 shape audio:", output2.shape)
        
        # return output2
        return output, output2, last_hidden 

    def compute_loss(self, batch_pred, y_labels):
        # Use Mean Squared Error loss for regression
        loss_fn = nn.MSELoss()
        loss = loss_fn(batch_pred, y_labels)
        return loss

    def create_optimizer(self, lr):
        # Optimizer (Adam) with the specified learning rate
        optimizer = optim.Adam(self.parameters(), lr=lr)
        return optimizer
