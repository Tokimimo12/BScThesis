import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)


    def forward(self, hidden_states, lengths):

        # print("hidden_states in attention:", hidden_states.shape)
        # print("lengths in attention:", lengths.shape)

        attn_scores = self.attn(hidden_states).squeeze(-1)  # [batch_size, seq_len]
        # print(f"attn_scores shape: {attn_scores.shape}")

        attn_weights = F.softmax(attn_scores, dim=1)  # [batch_size, seq_len]
        # print(f"attn_weights shape: {attn_weights.shape}")

        weighted_features = hidden_states * attn_weights.unsqueeze(-1)  # [batch_size, seq_len, hidden_dim]
        weighted_sum = weighted_features.sum(dim=1)  # [batch_size, hidden_dim]
        # print(f"weighted_sum shape: {weighted_sum.shape}")

        return weighted_sum, attn_weights
    
    # def forward(self, hidden_states):
    #     # hidden_states: [batch_size, seq_len, hidden_dim]
        
    #     print("hidden_states in attention:", hidden_states.shape)
    #     # Compute attention scores for each timestep
    #     attn_weights = self.attn(hidden_states)  # Shape: [batch_size, seq_len, 1]
    #     print("attn_weights in attention before softmax:", attn_weights.shape)

    #     attn_weights = torch.softmax(attn_weights, dim=1)  # Normalize over seq_len
    #     print("attn_weights in attention after softmax:", attn_weights.shape)

    #     # Weighted sum of hidden_states
    #     weighted_features = attn_weights * hidden_states  # Shape: [batch_size, seq_len, hidden_dim]
    #     print("weighted_features in attention:", weighted_features.shape)
    #     weighted_sum = torch.sum(weighted_features, dim=1)  # Shape: [batch_size, hidden_dim]
    #     print("weighted_sum in attention:", weighted_sum.shape)

    #     return weighted_sum  # Return attended features