import torch
import torch.nn as nn
import torch.optim as optim
from SingleEncoderModelAudio import SingleEncoderModelAudio
from SingleEncoderModelText import SingleEncoderModelText
from Attention import Attention

class EncoderMDREAttention(nn.Module):
    def __init__(self,
                 word2id,
                 encoder_size_audio,
                 num_layer_audio,
                 hidden_dim_audio,
                 dr_audio,
                 dic_size,
                 use_glove,
                 encoder_size_text,
                 num_layer_text,
                 hidden_dim_text,
                 dr_text: float,
                 output_size : int,
                 FUSION_TEHNIQUE = "concat",
                 word_to_vector_path=None
                 ):
        super(EncoderMDREAttention, self).__init__()

        # Audio model parameters
        self.encoder_size_audio = encoder_size_audio
        self.num_layer_audio = num_layer_audio
        self.hidden_dim_audio = hidden_dim_audio
        self.dr_audio = dr_audio

        # Text model parameters
        self.dic_size = dic_size
        self.use_glove = use_glove
        self.encoder_size_text = encoder_size_text
        self.num_layer_text = num_layer_text
        self.hidden_dim_text = hidden_dim_text
        self.dr_text = dr_text
        self.word2id = word2id
        self.word_to_vector_path = word_to_vector_path

        # Common parameters
        self.output_size = int(output_size)
        self.FUSION_TEHNIQUE = FUSION_TEHNIQUE
        
        if self.use_glove: 
            self.embed_dim = 300 
            print("GLOVE 300!!!")
        else: 
            print("\n----No glove embedding----\n")

        # Define attention layers
        self.attention_text = Attention(hidden_dim_text)
        self.attention_audio = Attention(hidden_dim_audio)

        # Initialize text and audio models
        self.text_model = SingleEncoderModelText(
            word2id=self.word2id,  
            dic_size=self.dic_size,
            use_glove=self.use_glove,
            encoder_size=self.encoder_size_text,
            num_layers=self.num_layer_text,
            hidden_dim=self.hidden_dim_text,
            dr=self.dr_text,
            output_size=self.output_size,
            word_to_vector_path=self.word_to_vector_path
        )
        self.audio_model = SingleEncoderModelAudio(
            input_size=self.encoder_size_audio,
            hidden_dim=self.hidden_dim_audio,
            num_layers=self.num_layer_audio,
            dropout_rate=self.dr_audio,
            output_size=self.output_size
        )

        # Define output projection layers
        combined_dim = hidden_dim_audio + hidden_dim_text
        self.fc = nn.Linear(combined_dim, output_size)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        self.text_weight = nn.Parameter(torch.randn(1))  # Learnable weight for text features
        self.audio_weight = nn.Parameter(torch.randn(1))  # Learnable weight for audio features

    def forward(self, text_inputs, audio_inputs, lengths, FUSION_TEHNIQUE = "concat"):
        batch_size = lengths.size(0)

        text_inputs = text_inputs.long()
        text_output, _, text_features_last_hidden = self.text_model(text_inputs, lengths) 
        audio_output, _, audio_features_last_hidden = self.audio_model(audio_inputs, lengths)  
        # print(f"text_output shape: {text_output.shape}")
        # print(f"text_last_hidden shape: {text_features_last_hidden.shape}")
        # print(f"audio_output2 shape: {audio_output.shape}")
        # print(f"audio_last_hidden shape: {audio_features_last_hidden.shape}")
        # Apply attention to text
        attended_text, attn_weights_text = self.attention_text(text_output, lengths)  # [batch_size, hidden_dim]
        # print(f"attended_text shape: {attended_text.shape}")
        # print(f"attn_weights_text shape: {attn_weights_text.shape}")
        attended_audio, attn_weights_audio = self.attention_audio(audio_output, lengths)  # [batch_size, hidden_dim]
        # print(f"attended_audio shape: {attended_audio.shape}")
        # print(f"attn_weights_audio shape: {attn_weights_audio.shape}")
        # print(self.FUSION_TEHNIQUE)
        if FUSION_TEHNIQUE == "concat":
            combined_features = torch.cat((attended_text, attended_audio), dim=1)
        elif FUSION_TEHNIQUE == "multiplication":        
            interaction_features = attended_text * attended_audio
            combined_features = torch.cat((attended_text, attended_audio, interaction_features), dim=1)
        elif FUSION_TEHNIQUE == "max":
            combined_features = torch.max((attended_text, attended_audio), dim=1)
        elif FUSION_TEHNIQUE == "weighted_sum":
            weighted_text = attended_text * self.text_weight
            weighted_audio = attended_audio * self.audio_weight
            combined_features = weighted_text + weighted_audio
        else:
            raise ValueError(f"Unsupported FUSION_TEHNIQUE: {FUSION_TEHNIQUE}")
        # Combine attended text and audio features
        # combined_features = torch.cat((attended_text, attended_audio), dim=1)  # [batch_size, 2 * hidden_dim]
        # print(f"combined_features shape: {combined_features.shape}")

        # Pass through fully connected layer
        final_features = self.fc(combined_features)  # [batch_size, hidden_dim]
        # print(f"final_features shape: {final_features.shape}")

        return final_features

    def compute_loss(self, logits, labels):
        return self.loss_fn(logits, labels)

    def create_optimizer(self, lr):
        return optim.Adam(self.parameters(), lr=lr)
