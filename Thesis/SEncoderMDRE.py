import torch
import torch.nn as nn
import torch.optim as optim
from SingleEncoderModelAudio import SingleEncoderModelAudio
from SingleEncoderModelText import SingleEncoderModelText


class EncoderMDRE(nn.Module):
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
                 word_to_vector_path=None
                 ):
        super(EncoderMDRE, self).__init__()

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
        # self.batch_size = batch_size
        # self.lr = lr
        self.output_size = int(output_size)
        
        if self.use_glove: 
            self.embed_dim = 300 
            print("GLOVE 300!!!")
        else: 
            print("\n----No glove embedding----\n")

        # Define sub-models
        self.audio_model = SingleEncoderModelAudio(
            input_size=self.encoder_size_audio,
            hidden_dim=self.hidden_dim_audio,
            num_layers=self.num_layer_audio,
            dropout_rate=self.dr_audio,
            output_size=self.output_size
        )

        print("\n----Audio model initialized----\n")

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

        print("\n----Text model initialized----\n")

        # Define output projection layers
        # self.output_layer = nn.Linear(
        #     (self.hidden_dim_audio // 2) + (self.hidden_dim_text // 2), self.output_size
        # )
        combined_dim = hidden_dim_audio + hidden_dim_text
        self.fc = nn.Linear(combined_dim, output_size)


        # Loss function
        self.loss_fn = nn.MSELoss()

        print("\n----Model architecture----\n")

    def forward(self, audio_inputs, text_inputs, lengths):
        batch_size = lengths.size(0)

        # Encode audio and text
        audio_features = self.audio_model(audio_inputs)  # [batch_size, audio_hidden_dim]
        text_features = self.text_model(text_inputs)     # [batch_size, text_hidden_dim]

        # Concatenate features
        combined_features = torch.cat((audio_features, text_features), dim=1)  # [batch_size, audio_hidden_dim + text_hidden_dim]

        # Output projection
        output = self.fc(combined_features)  # [batch_size, output_dim]
        return output

    def compute_loss(self, logits, labels):
        print("---loss computed---")
        return self.loss_fn(logits, labels)

    def create_optimizer(self):
        print("---optimizer created---")
        return optim.Adam(self.parameters(), lr=self.lr)

# Example usage
# Instantiate with appropriate parameters
# model = EncoderMDRE(batch_size=32, lr=0.001, encoder_size_audio=128, num_layer_audio=2,
#                     hidden_dim_audio=256, dr_audio=0.3, dic_size=10000, use_glove=1,
#                     encoder_size_text=128, num_layer_text=2, hidden_dim_text=256, 
#                     dr_text=0.3, n_category=10)
