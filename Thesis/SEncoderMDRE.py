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
                 FUSION_TEHNIQUE = "concat",
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
        self.FUSION_TEHNIQUE = FUSION_TEHNIQUE
        
        if self.use_glove: 
            self.embed_dim = 300 
            print("GLOVE 300!!!")
        else: 
            print("\n----No glove embedding----\n")


        # print(f"""\n----Text model initialized----: 
        #       word2id={word2id},  
        #       dic_size={dic_size},
        #       use_glove={use_glove},
        #       encoder_size={encoder_size_text},
        #       num_layers={num_layer_text},
        #       hidden_dim={hidden_dim_text},
        #       dr={dr_text},
        #       output_size={output_size},
        #       word_to_vector_path={word_to_vector_path}\n""")
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
        # print(f"Text model initialized")
        # Define sub-models
        # print(f"""\n----Audio model initialized---- 
        #       input_size= {encoder_size_audio}, 
        #       hidden_dim= {hidden_dim_audio}, 
        #       num_layers= {num_layer_audio}, 
        #       dropout_rate= {dr_audio}, 
        #       output_size = {output_size}\n""")
        self.audio_model = SingleEncoderModelAudio(
            input_size=self.encoder_size_audio,
            hidden_dim=self.hidden_dim_audio,
            num_layers=self.num_layer_audio,
            dropout_rate=self.dr_audio,
            output_size=self.output_size
        )
        # print(f"Audio model initialized")

        self.text_weight = nn.Parameter(torch.randn(1))  # Learnable weight for text features
        self.audio_weight = nn.Parameter(torch.randn(1))  # Learnable weight for audio features

        # Define output projection layers
        # self.output_layer = nn.Linear(
        #     (self.hidden_dim_audio // 2) + (self.hidden_dim_text // 2), self.output_size
        # )
        combined_dim = hidden_dim_audio + hidden_dim_text
        # print(f"\n----Combined dim: {combined_dim}----\n")
        self.fc = nn.Linear(combined_dim, output_size)

        self.tanh = nn.Tanh()

        # Loss function
        self.loss_fn = nn.MSELoss()

        # print("\n----Model architecture----\n")

    def forward(self, text_inputs, audio_inputs, lengths, FUSION_TEHNIQUE="concat"):
        batch_size = lengths.size(0)

        # Encode audio and text
        # print("text_inputs shape1:", text_inputs.shape)
        # print("audio_inputs shape1:", audio_inputs.shape)
        # print("lengths:", lengths)
        text_inputs = text_inputs.long()  # Convert to torch.LongTensor if not already
        _, _, text_features_last_hidden = self.text_model(text_inputs, lengths)     # [batch_size, text_hidden_dim]

        _, _, audio_features_last_hidden = self.audio_model(audio_inputs, lengths)  # [batch_size, audio_hidden_dim]
        # print("text_features shape2:", text_features_last_hidden.shape)

        # print("audio_features shape2:", audio_features_last_hidden.shape)

        
        # print("FUSION_TEHNIQUE:", self.FUSION_TEHNIQUE)
        
        if FUSION_TEHNIQUE == "concat":
            combined_features = torch.cat((text_features_last_hidden, audio_features_last_hidden), dim=1)
        elif FUSION_TEHNIQUE == "multiplication":        
            interaction_features = text_features_last_hidden * audio_features_last_hidden
            combined_features = torch.cat((text_features_last_hidden, audio_features_last_hidden, interaction_features), dim=1)
        elif FUSION_TEHNIQUE == "max":
            combined_features = torch.max((text_features_last_hidden, audio_features_last_hidden), dim=1)
        elif FUSION_TEHNIQUE == "weighted_sum":
            weighted_text = text_features_last_hidden * self.text_weight
            weighted_audio = audio_features_last_hidden * self.audio_weight
            combined_features = weighted_text + weighted_audio
        else:
            raise ValueError(f"Unsupported FUSION_TEHNIQUE: {FUSION_TEHNIQUE}")
        
        
        # print("combined_features shape:", combined_features.shape)    

        # Output projection
        output = self.fc(combined_features)  # [batch_size, output_dim]
        output2 = 3 * self.tanh(output)
        # print("output2 shape:", output2.shape)

        return output2

    def compute_loss(self, logits, labels):
        print("---loss computed---")
        return self.loss_fn(logits, labels)

    def create_optimizer(self, lr):
        print("---optimizer created---")
        return optim.Adam(self.parameters(), lr=lr)

# Example usage
# Instantiate with appropriate parameters
# model = EncoderMDRE(batch_size=32, lr=0.001, encoder_size_audio=128, num_layer_audio=2,
#                     hidden_dim_audio=256, dr_audio=0.3, dic_size=10000, use_glove=1,
#                     encoder_size_text=128, num_layer_text=2, hidden_dim_text=256, 
#                     dr_text=0.3, n_category=10)
