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
        
        self.gru = nn.GRU(input_size=self.input_size, 
                          hidden_size=self.hidden_dim,  
                          num_layers=self.num_layers, 
                          batch_first=True,  
                          dropout=self.dropout_rate if self.num_layers > 1 else 0)
        
        self.fc2 = nn.Linear(self.hidden_dim, self.output_size)
        
        self.softmax = nn.Softmax(dim=1)  # Softmax for classification
        

    def forward(self, x, lengths):
        batch_size = lengths.size(0)
        x = x.float()
        
        output, hidden = self.gru(x)  
        last_hidden = hidden[-1]  
        
        logits = self.fc2(last_hidden)  # Raw logits
        # output2 = self.softmax(logits) # Normalized probabilities
        
        return output, logits, last_hidden 


    def compute_loss(self, batch_pred_logits, y_labels):
        loss_fn = nn.CrossEntropyLoss()

        if y_labels.ndimension() > 1:  # Check if y_labels are one-hot encoded
            y_labels = torch.argmax(y_labels, dim=1)  # Convert to class indices

        loss = loss_fn(batch_pred_logits, y_labels)
        return loss
    
    def create_optimizer(self, lr):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        return optimizer