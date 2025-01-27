import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch import optim
import csv
import os

class SingleEncoderModelText(nn.Module):
    def __init__(self, word2id, dic_size, use_glove, encoder_size, num_layers, hidden_dim, dr, output_size, word_to_vector_path=None):
        super(SingleEncoderModelText, self).__init__()
        
        # Parameters
        self.word2id = word2id
        self.dic_size = dic_size
        self.use_glove = use_glove
        self.encoder_size = encoder_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dr = dr
        self.output_size = output_size
        # self.FUSION_TEHNIQUE = FUSION_TEHNIQUE
        
        # Embedding dimensions
        if self.use_glove: self.embed_dim = 300  
        
        # Embedding layer
        self.embedding = nn.Embedding(self.dic_size, self.embed_dim)
        
        # If using GloVe, load embeddings
        if self.use_glove and word_to_vector_path:
            self._load_pretrained_embeddings(word2id, word_to_vector_path)
        
        # GRU layer
        self.gru = nn.GRU(input_size=self.embed_dim, 
                          hidden_size=self.hidden_dim, 
                          num_layers=self.num_layers, 
                          dropout=self.dr, 
                          batch_first=True)
        
        # Fully connected output layer
        self.fc2 = nn.Linear(self.hidden_dim, self.output_size)
        
        # Tanh activation function to scale outputs between -3 and 3
        self.tanh = nn.Tanh()

    def _load_pretrained_embeddings(self, word2id, word_to_vector_path):
        """
        Load pretrained GloVe embeddings and set them in the embedding layer.
        
        Args:
            word2id (dict): A dictionary mapping words to their indices in the model vocabulary.
            word_to_vector_path (str): Path to the CSV file containing GloVe embeddings.
        """
        # Read GloVe vectors
        print(f"Loading GloVe embeddings from {word_to_vector_path}...")
        glove_df = pd.read_csv(word_to_vector_path, header=None)
        
        # Separate words and vectors
        words_glove = glove_df.iloc[:, 0].values
        vectors_glove = glove_df.iloc[:, 1:].values
        glove_dim = vectors_glove.shape[1]
        
        # Validate dimensions
        if self.embed_dim != glove_dim:
            raise ValueError(f"Embedding dimension mismatch: Model({self.embed_dim}) vs GloVe({glove_dim}).")
        
        # Create a word-to-index mapping for GloVe
        word_to_idx_glove = {word: idx for idx, word in enumerate(words_glove)}
        
        # Initialize weight matrix with random values
        embedding_matrix = np.random.uniform(-0.05, 0.05, (self.dic_size, self.embed_dim))
        
        # Assign GloVe vectors to the weight matrix for known words
        found_words = 0
        for word, idx in word2id.items():
            if word in word_to_idx_glove:  # Check if word exists in GloVe
                glove_idx = word_to_idx_glove[word]
                embedding_matrix[idx] = vectors_glove[glove_idx]
                found_words += 1
            elif word in ['unk', 'pad']:  # Initialize `unk` and `pad` with random values
                embedding_matrix[idx] = np.random.uniform(-0.05, 0.05, self.embed_dim)
                print(f"Randomly initialized '{word}' embedding.")
        
        print(f"Found {found_words}/{self.dic_size} words in the GloVe embeddings.")
        
        # Set weights in the embedding layer
        self.embedding.weight.data = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding.weight.requires_grad = False  # Freeze embeddings to prevent updates
        print("Embedding layer weights set.")


    def forward(self, x, lengths):
        batch_size = lengths.size(0)
        x = x.float()

        if (x.min() < 0 or x.max() >= self.dic_size):
            raise ValueError(
                f"Input indices out of range! Min index: {x.min()}, Max index: {x.max()}, Vocabulary size: {self.dic_size}"
            )
        x = x.long()  
        embedded = self.embedding(x)  # [batch_size, seq_length] -> [batch_size, seq_length, embed_dim]
        # print("embedded shape text:", embedded.shape)
        output, hidden = self.gru(embedded)  # gru_out: [batch_size, seq_length, hidden_dim], hidden: [num_layers, batch_size, hidden_dim]
        # print("output shape text:", output.shape)
        # print("hidden shape text:", hidden.shape)
        last_hidden = hidden[-1]  # Shape: [batch_size, hidden_dim]
        # print("last_hidden shape text:", last_hidden.shape)
        output1 = self.fc2(last_hidden)  # [batch_size, output_size]
        # print("output1 shape text:", output1.shape)
        output2 = 3 * self.tanh(output1)
        # return output2
        # print("output2 shape text:", output2.shape)

        return output, output2, last_hidden
    
    
    def compute_loss(self, batch_pred, y_labels):
        loss_fn = nn.MSELoss()
        loss = loss_fn(batch_pred, y_labels)
        return loss

    def create_optimizer(self, lr):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        return optimizer


    def check_word_embedding(self, word, word2id, glove_path):
        # Read GloVe vectors
        glove_df = pd.read_csv(glove_path, header=None)
        words_glove = glove_df.iloc[:, 0].values
        vectors_glove = glove_df.iloc[:, 1:].values
        word_to_idx_glove = {word: idx for idx, word in enumerate(words_glove)}
        
        # Retrieve model embedding
        word_idx = word2id.get(word)
        if word_idx is None:
            print(f"Word '{word}' not found in word2id.")
            return
        
        model_embedding = self.embedding.weight.data[word_idx].cpu().numpy()
        
        # Retrieve GloVe embedding
        glove_idx = word_to_idx_glove.get(word)
        if glove_idx is None:
            print(f"Word '{word}' not found in GloVe.")
            return
        
        glove_embedding = vectors_glove[glove_idx]
        
        # Compare embeddings
        print(f"Model embedding for '{word}': {model_embedding}")
        print(f"GloVe embedding for '{word}': {glove_embedding}")
        if np.allclose(model_embedding, glove_embedding):
            print("Embeddings match!")
        else:
            print("Embeddings do not match!")

    def check_special_tokens(self, word2id):
        for special_token in ['<unk>', '<pad>']:
            idx = word2id.get(special_token)
            if idx is not None:
                embedding = self.embedding.weight.data[idx].cpu().numpy()
                print(f"Embedding for '{special_token}': {embedding}")
            else:
                print(f"Special token '{special_token}' not found in word2id.")
                new_idx = len(word2id)
                word2id[special_token] = new_idx
                # Initialize the embedding for the new token (optional, based on use case)
                self.embedding.weight.data = torch.cat(
                    [self.embedding.weight.data, torch.zeros(1, self.embedding.weight.size(1))], dim=0
                )
                print(f"Special token '{special_token}' added with index {new_idx}.")

    def verify_vocab_embeddings(model, word2id):
        for word, idx in word2id.items():
            embedding = model.embedding.weight.data[idx].cpu().numpy()
            if np.any(np.isnan(embedding)):
                print(f"Embedding for word '{word}' contains NaN values!")
                return False
        print("All vocabulary embeddings are valid!")
        return True



    def check_embedding_layer(self, word2id, glove_path):
        print("### Embedding Layer Check ###")
        print(f"Embedding weights shape: {self.embedding.weight.data.shape}")
        print("\nChecking a few words...")
        self.check_word_embedding('example', word2id, glove_path)
        self.check_word_embedding('unk', word2id, glove_path)
        print("\nChecking special tokens...")
        self.check_special_tokens(word2id)
        print("\nValidating all embeddings...")
        self.verify_vocab_embeddings(word2id)
        print("\nEmbedding layer frozen status:")
        print("Frozen" if not self.embedding.weight.requires_grad else "Not frozen")



    def verify_embedding(self, word, word_to_vector_path):
        # Load GloVe data
        glove_df = pd.read_csv(word_to_vector_path, header=None)
        words = glove_df.iloc[:, 0].values
        vectors = glove_df.iloc[:, 1:].values

        word_to_idx = {word: idx for idx, word in enumerate(words)}

        # Check if the word is in GloVe
        if word in word_to_idx:
            idx = word_to_idx[word]
            glove_vector = vectors[idx]
            model_vector = self.embedding.weight[idx].detach().numpy()
            print(f"GloVe vector for '{word}': {glove_vector}")
            print(f"Model vector for '{word}': {model_vector}")
            return np.allclose(glove_vector, model_vector)
        else:
            print(f"Word '{word}' not found in GloVe!")
            return False
        

    def align_embeddings(self, word2id, glove_path, embed_dim):
        """
        Aligns the word2id dictionary with GloVe embeddings.

        Args:
            word2id (dict): A dictionary mapping words to indices.
            glove_path (str): Path to the GloVe word-to-vector CSV file.
            embed_dim (int): Dimensionality of the GloVe embeddings.

        Returns:
            tuple: (embedding_weights, word_to_idx_glove, vectors_glove, missing_words, common_words)
        """

        # Load GloVe data
        glove_df = pd.read_csv(glove_path, header=None)
        words_glove = glove_df.iloc[:, 0].str.lower().str.strip()  # Normalize words
        vectors_glove = glove_df.iloc[:, 1:].to_numpy()  # Extract vectors as numpy array

        # Map words to indices in GloVe
        word_to_idx_glove = {word: idx for idx, word in enumerate(words_glove)}

        # Find common and missing words
        common_words = set(word2id.keys()) & set(word_to_idx_glove.keys())
        missing_words = set(word2id.keys()) - set(word_to_idx_glove.keys())

        # Initialize embedding weights
        vocab_size = len(word2id)
        embedding_weights = np.random.uniform(-0.1, 0.1, (vocab_size, embed_dim))

        # Populate embeddings for common words
        for word in common_words:
            idx = word2id[word]
            glove_idx = word_to_idx_glove[word]
            embedding_weights[idx] = vectors_glove[glove_idx]

        return embedding_weights, word_to_idx_glove, vectors_glove, missing_words, common_words


    def save_embeddings_to_csv(self, word2id, embedding_weights, output_file):
        """
        Saves the words and their corresponding embedding weights to a CSV file.

        Args:
        - word2id (dict): Mapping of words to their indices.
        - embedding_weights (numpy.ndarray): The embedding matrix (vocab_size x embed_dim).
        - output_file (str): Path to the output CSV file.
        """

        directory = os.path.dirname(output_file)
        if directory:  # Only try to create the directory if it exists in the path
            os.makedirs(directory, exist_ok=True)
        # Reverse mapping: id2word
        id2word = {idx: word for word, idx in word2id.items()}
        
        # Save to CSV
        with open(output_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
                        
            # Write word and embeddings
            for idx, embedding in enumerate(embedding_weights):
                word = id2word.get(idx, f"Index_{idx}")  # Handle cases where idx doesn't have a word
                writer.writerow([word] + embedding.tolist())

        print(f"Embeddings saved to {output_file}")

    def verify_all_embeddings(self, word2id, word_to_vector_path, output_file):
        """
        Verifies if all GloVe embeddings have been correctly loaded into the embedding layer,
        initializes missing words (e.g., `unk`, `pad`) with random values, and saves embeddings.
        """
        if not isinstance(word2id, dict):
            raise ValueError("word2id must be a dictionary mapping words to indices.")

        # Fix potential list indices in word2id
        word2id = {word: (indices[0] if isinstance(indices, list) else indices) for word, indices in word2id.items()}

        # Reverse the mapping
        id2word = {idx: word for word, idx in word2id.items()}

        # Load GloVe data
        glove_df = pd.read_csv(word_to_vector_path, header=None)
        words = glove_df.iloc[:, 0].values
        vectors = glove_df.iloc[:, 1:].values
        embed_dim = vectors.shape[1]

        word_to_idx = {word: idx for idx, word in enumerate(words)}
        embedding_weights = self.embedding.weight.data.cpu().numpy()

        missing_words = []
        mismatched_words = []
        correctly_loaded = 0

        # Iterate over all words in vocabulary
        for idx in range(self.dic_size):
            word = id2word.get(idx, None)  # Look up the word corresponding to the index
            if word is None:
                print(f"Index {idx} is missing in id2word mapping.")
                missing_words.append(f"Index_{idx}")
                continue

            if word in word_to_idx:  # Check if the word exists in GloVe
                glove_idx = word_to_idx[word]
                glove_vector = vectors[glove_idx]
                model_vector = embedding_weights[idx]

                if np.allclose(glove_vector, model_vector):
                    correctly_loaded += 1
                else:
                    mismatched_words.append(word)
            else:
                missing_words.append(word)

        # Randomly initialize missing words
        # for word in missing_words:
        #     if word in ['unk', 'pad']:
        #         idx = word2id[word]
        #         embedding_weights[idx] = np.random.uniform(-0.1, 0.1, embed_dim)

        # Update the embedding matrix with the new weights
        self.embedding.weight.data = torch.tensor(embedding_weights)

        # Save embeddings to a CSV file
        self.save_embeddings_to_csv(word2id, embedding_weights, output_file)

        total_words = self.dic_size
        print(f"Verification Report:")
        print(f"Total words in vocabulary: {total_words}")
        print(f"Correctly loaded embeddings: {correctly_loaded}")
        print(f"Words not found in GloVe: {len(missing_words)}")
        print(f"Words with mismatched embeddings: {len(mismatched_words)}")
        print(f"Common words: {correctly_loaded}")

        return {
            "total_words": total_words,
            "correctly_loaded": correctly_loaded,
            "missing_words": missing_words,
            "mismatched_words": mismatched_words,
        }


            
            