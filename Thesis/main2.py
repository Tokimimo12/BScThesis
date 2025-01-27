from preprocessing import Preprocessing
from visualisation import Visualisation
from collate import CollateBatch
from SEncoderMDRE import EncoderMDRE
from SEncoderMDREAtt import EncoderMDREAttention
from SingleEncoderModelAudio import SingleEncoderModelAudio
from SingleEncoderModelText import SingleEncoderModelText
from traintest import TrainTest
from constants import SDK_PATH, DATA_PATH, WORD_EMB_PATH, CACHE_PATH

import torch
from torch.utils.data import DataLoader
from itertools import product


# Define hyperparameter grid
hyperparameter_grid = {
    "MODEL_TYPE": ["MDRE", "MDREAttention", "SingleEncoderModelAudio", "SingleEncoderModelText"],
    "FUSION_TEHNIQUE": ["concat", "multiplication", "max", "weighted_sum"],

    "hidden_dim": [64, 128
                        #  256, 
                        #  512
                         ],
    "num_layers": [2, 4 
                        #  8
                         ],
    "dropout": [0.3, 0.5],
    # "text_dropout": [0.3, 0.5],
    "batch_size": [32, 64],
    "patience": [8, 16],

  
    # "audio_hidden_dim": [128],
    # "audio_num_layers": [2],
    # "audio_dropout": [0.5],
    # "text_hidden_dim": [128],
    # "text_num_layers": [2],
    # "text_dropout": [0.5],
    # "batch_size": [56],
    # "MODEL_TYPE": ["MDRE", "MDREAttention", "SingleEncoderModelAudio", "SingleEncoderModelText"],

    # "FUSION_TEHNIQUE": ["concat", "multiplication", "max", "weighted_sum"],
    # "patience": [8]
}

def preprocessing():
        # Initialize Preprocessing instance
    preprocessing = Preprocessing()
    visualisation = Visualisation()

    # Initialize and preprocess data
    preprocessing.initialize_sdk()
    datasetMD = preprocessing.setup_data()
    dataset, visual_field, acoustic_field, text_field, wordvectors_field = preprocessing.load_features_and_save(datasetMD)
    dataset, label_field = preprocessing.align_labels(dataset)
    train_split, dev_split, test_split = preprocessing.split_data(datasetMD)
    train, dev, test, word2id = preprocessing.preprocess_data_modified(dataset, visual_field, acoustic_field, text_field, wordvectors_field, label_field, train_split, dev_split, test_split)
    # visualisation.plot_sentiment_histogram(csv_file="labels.csv")

    return train, dev, test, word2id 
    

def build(train, dev, test, word2id, params):
    """Build, train, and evaluate the model for specific hyperparameters."""

    # Extract hyperparameters from params
    MODEL_TYPE = params["MODEL_TYPE"]
    FUSION_TEHNIQUE = params["FUSION_TEHNIQUE"]
    hidden_dim = params["hidden_dim"]
    num_layers = params["num_layers"]
    dropout = params["dropout"]
    batch_size = params["batch_size"]
    
    patience = params["patience"]
    # num_trials = params["num_trials"]

    print(f"""MODEL_TYPE: {MODEL_TYPE}, 
          FUSION_TEHNIQUE: {FUSION_TEHNIQUE}, 
          hidden_dim: {hidden_dim}, 
          num_layers: {num_layers}, 
          dropout: {dropout}, 
          
          batch_size: {batch_size}, 
          patience: {patience}""")



    print("words: ", word2id)

    # Model parameters
    audio_input_size = 74
    dic_size = len(word2id)

    use_glove = True
    text_encoder_size = 300
    output_size = 1

    pad_value= word2id['<pad>']

    num_trials = 5



    # Select and initialize the model based on MODEL_TYPE
    if MODEL_TYPE == "SingleEncoderModelAudio":
        model = SingleEncoderModelAudio(
            input_size=audio_input_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout,
            # FUSION_TEHNIQUE=FUSION_TEHNIQUE,
            output_size=output_size
        )

    elif MODEL_TYPE == "SingleEncoderModelText":
        model = SingleEncoderModelText(
            word2id=word2id,
            dic_size=dic_size,
            use_glove=use_glove,
            encoder_size=text_encoder_size,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dr=dropout,
            output_size=output_size,
            # FUSION_TEHNIQUE=FUSION_TEHNIQUE,
            word_to_vector_path='/home1/s4680340/BScThesis/Thesis/word_to_vector_mapping.csv'
        )

    elif MODEL_TYPE == "MDRE":
        model = EncoderMDRE(
            word2id=word2id,
            encoder_size_audio=audio_input_size,
            num_layer_audio=num_layers,
            hidden_dim_audio=hidden_dim,
            dr_audio=dropout,
            dic_size=dic_size,
            use_glove=use_glove,
            encoder_size_text=text_encoder_size,
            num_layer_text=num_layers,
            hidden_dim_text=hidden_dim,
            dr_text=dropout,
            output_size=output_size,
            FUSION_TEHNIQUE=FUSION_TEHNIQUE,
            word_to_vector_path='/home1/s4680340/BScThesis/Thesis/word_to_vector_mapping.csv'
        )

    elif MODEL_TYPE == "MDREAttention":
        model = EncoderMDREAttention(
            word2id=word2id,
            encoder_size_audio=audio_input_size,
            num_layer_audio=num_layers,
            hidden_dim_audio=hidden_dim,
            dr_audio=dropout,
            dic_size=dic_size,
            use_glove=use_glove,
            encoder_size_text=text_encoder_size,
            num_layer_text=num_layers,
            hidden_dim_text=hidden_dim,
            dr_text=dropout,
            output_size=output_size,
            FUSION_TEHNIQUE=FUSION_TEHNIQUE,
            word_to_vector_path='/home1/s4680340/BScThesis/Thesis/word_to_vector_mapping.csv'
        )

    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    pad_value= word2id['<pad>']

    collate = CollateBatch(MODEL_TYPE, pad_value)

    cuda_avalilable = torch.cuda.is_available()

    train_loader = DataLoader(train, shuffle=True, batch_size=batch_size, collate_fn=lambda batch: collate.collate_batch(batch))
    dev_loader = DataLoader(dev, shuffle=False, batch_size=batch_size * 3, collate_fn=lambda batch: collate.collate_batch(batch))
    test_loader = DataLoader(test, shuffle=False, batch_size=batch_size * 3, collate_fn=lambda batch: collate.collate_batch(batch))
    
    trainer = TrainTest(MODEL_TYPE=MODEL_TYPE, 
                        model=model, 
                        FUSION_TEHNIQUE=FUSION_TEHNIQUE, 
                        batch_size=batch_size,
                        train_loader=train_loader, 
                        dev_loader=dev_loader, 
                        test_loader=test_loader, 
                        cuda_available=cuda_avalilable, 
                        max_epoch=1000, 
                        patience = patience, 
                        num_trials = num_trials, 
                        grad_clip_value=1.0)

    # Train the model
    print("Starting training...")
    trained_model = trainer.train_model()

    # Test the model
    print("Starting testing...")
    metrics = trainer.test_model_classification()

    return trained_model, metrics
    
def grid_search(train, dev, test, word2id ):
    """Perform grid search to find the best hyperparameter configuration."""
    keys, values = zip(*hyperparameter_grid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]

    # Initialize variables to track the best hyperparameters and metrics for each model type
    best_mdre_concat_params = None
    best_mdre_concat_metrics = None

    best_mdre_multiplication_params = None
    best_mdre_multiplication_metrics = None

    best_mdre_max_params = None
    best_mdre_max_metrics = None

    best_mdre_weighted_sum_params = None
    best_mdre_weighted_sum_metrics = None

    best_mdre_attention_concat_params = None
    best_mdre_attention_concat_metrics = None
    
    best_mdre_attention_multiplication_params = None
    best_mdre_attention_multiplication_metrics = None

    best_mdre_attention_max_params = None
    best_mdre_attention_max_metrics = None

    best_mdre_attention_weighted_sum_params = None
    best_mdre_attention_weighted_sum_metrics = None

    best_audio_params = None
    best_audio_metrics = None

    best_text_params = None
    best_text_metrics = None



    # Iterate through all combinations of hyperparameters
    for params in combinations:
        print(f"Testing combination: {params}")

        # Train and test the model with the current hyperparameters
        _, metrics = build(train, dev, test, word2id ,params)

        # Update the best hyperparameters and metrics for each model type
        if params["MODEL_TYPE"] == "MDRE" and params["FUSION_TEHNIQUE"] == "concat":
            if best_mdre_concat_metrics is None or metrics["accuracy"] > best_mdre_concat_metrics["accuracy"]:
                best_mdre_concat_params = params
                best_mdre_concat_metrics = metrics

        elif params["MODEL_TYPE"] == "MDRE" and params["FUSION_TEHNIQUE"] == "multiplication":
            if best_mdre_multiplication_metrics is None or metrics["accuracy"] > best_mdre_multiplication_metrics["accuracy"]:
                best_mdre_multiplication_params = params
                best_mdre_multiplication_metrics = metrics

        elif params["MODEL_TYPE"] == "MDRE" and params["FUSION_TEHNIQUE"] == "max":
            if best_mdre_max_metrics is None or metrics["accuracy"] > best_mdre_max_metrics["accuracy"]:
                best_mdre_max_params = params
                best_mdre_max_metrics = metrics

        elif params["MODEL_TYPE"] == "MDRE" and params["FUSION_TEHNIQUE"] == "weighted_sum":
            if best_mdre_weighted_sum_metrics is None or metrics["accuracy"] > best_mdre_weighted_sum_metrics["accuracy"]:
                best_mdre_weighted_sum_params = params
                best_mdre_weighted_sum_metrics = metrics

        elif params["MODEL_TYPE"] == "MDREAttention" and params["FUSION_TEHNIQUE"] == "concat":
            if best_mdre_attention_concat_metrics is None or metrics["accuracy"] > best_mdre_attention_concat_metrics["accuracy"]:
                best_mdre_attention_concat_params = params
                best_mdre_attention_concat_metrics = metrics
        
        elif params["MODEL_TYPE"] == "MDREAttention" and params["FUSION_TEHNIQUE"] == "multiplication":
            if best_mdre_attention_multiplication_metrics is None or metrics["accuracy"] > best_mdre_attention_multiplication_metrics["accuracy"]:
                best_mdre_attention_multiplication_params = params
                best_mdre_attention_multiplication_metrics = metrics
        
        elif params["MODEL_TYPE"] == "MDREAttention" and params["FUSION_TEHNIQUE"] == "max":
            if best_mdre_attention_max_metrics is None or metrics["accuracy"] > best_mdre_attention_max_metrics["accuracy"]:
                best_mdre_attention_max_params = params
                best_mdre_attention_max_metrics = metrics

        elif params["MODEL_TYPE"] == "MDREAttention" and params["FUSION_TEHNIQUE"] == "weighted_sum":
            if best_mdre_attention_weighted_sum_metrics is None or metrics["accuracy"] > best_mdre_attention_weighted_sum_metrics["accuracy"]:
                best_mdre_attention_weighted_sum_params = params
                best_mdre_attention_weighted_sum_metrics = metrics

        elif params["MODEL_TYPE"] == "SingleEncoderModelAudio":
            if best_audio_metrics is None or metrics["accuracy"] > best_audio_metrics["accuracy"]:
                best_audio_params = params
                best_audio_metrics = metrics

        elif params["MODEL_TYPE"] == "SingleEncoderModelText":
            if best_text_metrics is None or metrics["accuracy"] > best_text_metrics["accuracy"]:
                best_text_params = params
                best_text_metrics = metrics

    # Print the best hyperparameters and metrics for each model type
    print("\nBest MDRE concat model:")
    print(f"Hyperparameters: {best_mdre_concat_params}")
    print(f"Metrics: {best_mdre_concat_metrics}")

    print("\nBest MDRE multiplication model:")
    print(f"Hyperparameters: {best_mdre_multiplication_params}")
    print(f"Metrics: {best_mdre_multiplication_metrics}")

    print("\nBest MDRE max model:")
    print(f"Hyperparameters: {best_mdre_max_params}")
    print(f"Metrics: {best_mdre_max_metrics}")

    print("\nBest MDRE weighted_sum model:")
    print(f"Hyperparameters: {best_mdre_weighted_sum_params}")
    print(f"Metrics: {best_mdre_weighted_sum_metrics}")

    print("\nBest MDREAttention concat model:")
    print(f"Hyperparameters: {best_mdre_attention_concat_params}")
    print(f"Metrics: {best_mdre_attention_concat_metrics}")

    print("\nBest MDREAttention multiplication model:")
    print(f"Hyperparameters: {best_mdre_attention_multiplication_params}")
    print(f"Metrics: {best_mdre_attention_multiplication_metrics}")

    print("\nBest MDREAttention max model:")
    print(f"Hyperparameters: {best_mdre_attention_max_params}")
    print(f"Metrics: {best_mdre_attention_max_metrics}")

    print("\nBest MDREAttention weighted_sum model:")
    print(f"Hyperparameters: {best_mdre_attention_weighted_sum_params}")
    print(f"Metrics: {best_mdre_attention_weighted_sum_metrics}")

    print("\nBest SingleEncoderModelAudio model:")
    print(f"Hyperparameters: {best_audio_params}")
    print(f"Metrics: {best_audio_metrics}")

    print("\nBest SingleEncoderModelText model:")
    print(f"Hyperparameters: {best_text_params}")
    print(f"Metrics: {best_text_metrics}")


if __name__ == "__main__":
    
    train, dev, test, word2id = preprocessing()

    grid_search(train, dev, test, word2id)
