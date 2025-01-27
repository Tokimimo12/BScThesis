import torch
from torch.nn.utils.rnn import pad_sequence

class CollateBatch:
    """
    A class to handle variable-length sequences in a batch.
    """
    def __init__(self, model_type, pad_value):
        """
        Initialize the CollateBatch class.

        Args:
            model_type (str): The type of model (e.g., 'MDREAttention', 'MDRE', etc.).
            pad_value (int): The padding value to use for text sequences.
        """
        self.MODEL_TYPE = model_type
        self.pad_value = pad_value

    def collate_batch(self, batch):
        """
        Collate function to process a batch of variable-length sequences.

        Args:
            batch (list): A list of samples, where each sample is a tuple (inputs, labels).

        Returns:
            tuple: Processed sentences, acoustic features, labels, and lengths.
        """
        batch = sorted(batch, key=lambda x: len(x[0][0]), reverse=True)
        labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0).float()

        # sentences = None
        # acoustic = None


        if self.MODEL_TYPE in ["MDREAttention", "MDRE"]:
            sentences = pad_sequence(
                [torch.LongTensor(sample[0][0]) for sample in batch],
                padding_value=self.pad_value,
                batch_first=True
            )
            acoustic = pad_sequence(
                [torch.FloatTensor(sample[0][2]) for sample in batch],
                batch_first=True
            )
            lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])
            return sentences, acoustic, labels, lengths
        
        elif self.MODEL_TYPE == "SingleEncoderModelAudio":
            # labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0).float()
            acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch], batch_first=True)
            
            lengths = torch.LongTensor([sample[0][2].shape[0] for sample in batch])
            return acoustic, labels, lengths

        elif self.MODEL_TYPE == "SingleEncoderModelText":
            sentences = pad_sequence(
                [torch.LongTensor(sample[0][0]) for sample in batch],
                padding_value=self.pad_value,
                batch_first=True
            )
            lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])
            return sentences, labels, lengths

        
