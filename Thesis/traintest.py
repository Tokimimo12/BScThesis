import torch
from torch import cuda
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch import optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TrainTest:
    def __init__(self, MODEL_TYPE, model, FUSION_TEHNIQUE, batch_size, train_loader, dev_loader, test_loader, cuda_available: bool = True, max_epoch=1000, patience=8, num_trials=3, grad_clip_value=1.0):
        self.model = model
        self.MODEL_TYPE = MODEL_TYPE
        self.batch_size = batch_size
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.max_epoch = max_epoch
        self.patience = patience
        self.num_trials = num_trials
        self.grad_clip_value = grad_clip_value
        self.cuda_available = cuda_available and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_available else "cpu")
        print(f"Using device: {self.device}")
        self.FUSION_TEHNIQUE = FUSION_TEHNIQUE
        print(f"Using fusion technique: {self.FUSION_TEHNIQUE}")

    def train_model(self):
        torch.manual_seed(123)
        torch.cuda.manual_seed_all(123)
        print("self.cuda_available:", self.cuda_available)
        if self.cuda_available:
            print("self.cuda_available:", self.cuda_available)
            torch.cuda.manual_seed_all(123)

        print("self.device:", self.device)

        optimizer = self.model.create_optimizer(lr=0.001)
        self.model.to(self.device)

        criterion = nn.MSELoss(reduction='sum')
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        lr_scheduler.step()

        best_valid_loss = float('inf')
        curr_patience = self.patience
        train_losses, valid_losses = [], []

        for e in range(self.max_epoch):
            self.model.train()
            train_iter = tqdm(self.train_loader)
            train_loss = 0.0

            for batch in train_iter:
                self.model.zero_grad()
                # print("MODEL_TYPE:", self.MODEL_TYPE)

                if self.MODEL_TYPE == "MDREAttention" or self.MODEL_TYPE == "MDRE":
                    t, a, y, l = batch
                    t, a, y, l = t.to(self.device), a.to(self.device), y.to(self.device), l.to(self.device)
                    # y_tilde = self.model(t, a, l, self.FUSION_TEHNIQUE)
                    y_tilde = self.model(t, a, l)
                    
                elif self.MODEL_TYPE == "SingleEncoderModelAudio":
                    a, y, l = batch
                    a, y, l = a.to(self.device), y.to(self.device), l.to(self.device)
                    _, y_tilde, _ = self.model(a, l)

                
                elif self.MODEL_TYPE == "SingleEncoderModelText":
                    t, y, l = batch
                    t, y, l = t.to(self.device), y.to(self.device), l.to(self.device)
                    _, y_tilde, _ = self.model(t, l)

                loss = criterion(y_tilde, y)
                loss.backward()
                torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], self.grad_clip_value)
                optimizer.step()

                train_iter.set_description(f"Epoch {e}/{self.max_epoch}, current batch loss: {round(loss.item()/self.batch_size, 4)}")

                train_loss += loss.item()


            train_loss /= len(self.train_loader)
            train_losses.append(train_loss)
            print(f"Training loss: {round(train_loss, 4)}")

            self.model.eval()

            with torch.no_grad():
                valid_loss = 0.0
                for batch in self.dev_loader:
                    self.model.zero_grad()
                    if self.MODEL_TYPE == "MDREAttention" or self.MODEL_TYPE == "MDRE":
                        t, a, y, l = batch
                        t, a, y, l = t.to(self.device), a.to(self.device), y.to(self.device), l.to(self.device)
                        # y_tilde = self.model(t, a, l, self.FUSION_TEHNIQUE)
                        y_tilde = self.model(t, a, l)
                    
                    elif self.MODEL_TYPE == "SingleEncoderModelAudio":
                        a, y, l = batch
                        a, y, l = a.to(self.device), y.to(self.device), l.to(self.device)
                        _, y_tilde, _ = self.model(a, l)
                    
                    elif self.MODEL_TYPE == "SingleEncoderModelText":
                        t, y, l = batch
                        t, y, l = t.to(self.device), y.to(self.device), l.to(self.device)
                        _, y_tilde, _ = self.model(t, l)
                        
                    loss = criterion(y_tilde, y)
                    valid_loss += loss.item()

            valid_loss /= len(self.dev_loader)
            valid_losses.append(valid_loss)
            print(f"Validation loss: {round(valid_loss, 4)}")

            if valid_loss <= best_valid_loss:
                best_valid_loss = valid_loss
                print("Found new best model on dev set!")

                model_filename = f'model{self.MODEL_TYPE}_{self.FUSION_TEHNIQUE}.std'
                optim_filename = f'optim{self.MODEL_TYPE}_{self.FUSION_TEHNIQUE}.std'
                torch.save(self.model.state_dict(), model_filename)
                torch.save(optimizer.state_dict(), optim_filename)
                curr_patience = self.patience
            else:
                curr_patience -= 1
                if curr_patience <= -1:
                    self.num_trials -= 1
                    curr_patience = self.patience
                    model_filename = f'model{self.MODEL_TYPE}_{self.FUSION_TEHNIQUE}.std'
                    optim_filename = f'optim{self.MODEL_TYPE}_{self.FUSION_TEHNIQUE}.std'
                    self.model.load_state_dict(torch.load(model_filename))
                    optimizer.load_state_dict(torch.load(optim_filename))
                    lr_scheduler.step()
                    print(f"Current learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")

            if self.num_trials <= 0:
                print("Early stopping.")
                break

        return self.model


    def test_model_classification(self):
        self.model.to(self.device)
        model_filename = f'model{self.MODEL_TYPE}_{self.FUSION_TEHNIQUE}.std'
        self.model.load_state_dict(torch.load(model_filename))
        print("Model loaded successfully!")

        y_true, y_pred = [], []
        self.model.eval()
        criterion = nn.MSELoss(reduction='sum')

        with torch.no_grad():
            test_loss = 0.0
            # print("self.test_loader:", self.test_loader.type)
            for batch in self.test_loader:
                if self.MODEL_TYPE == "MDREAttention" or self.MODEL_TYPE == "MDRE":
                    t, a, y, l = batch
                    t, a, y, l = t.to(self.device), a.to(self.device), y.to(self.device), l.to(self.device)
                    # y_tilde = self.model(t, a, l, self.FUSION_TEHNIQUE)
                    y_tilde = self.model(t, a, l)

                elif self.MODEL_TYPE == "SingleEncoderModelAudio":
                    a, y, l = batch
                    a, y, l = a.to(self.device), y.to(self.device), l.to(self.device)
                    output1, y_tilde, last_hidden = self.model(a, l)
                
                elif self.MODEL_TYPE == "SingleEncoderModelText":
                    t, y, l = batch
                    t, y, l = t.to(self.device), y.to(self.device), l.to(self.device)
                    output1text, y_tilde, last_hidden_text = self.model(t, l)
                    
                loss = criterion(y_tilde, y)
                test_loss += loss.item()

                y_true.append(y.detach().cpu().numpy())
                y_pred.append(y_tilde.detach().cpu().numpy())

        avg_test_loss = test_loss / len(self.test_loader)
        print(f"Test set performance (Average Loss): {avg_test_loss}")

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)

        y_true_categories = [self.convert_to_sentiment_category(score) for score in y_true]
        y_pred_categories = [self.convert_to_sentiment_category(score) for score in y_pred]

        metrics = self.evaluate(y_true_categories, y_pred_categories)
        print(f"Accuracy: {metrics['accuracy']}")
        print(f"Precision: {metrics['precision']}")
        print(f"Recall: {metrics['recall']}")
        print(f"F1 Score: {metrics['f1_score']}")

        return metrics

    @staticmethod
    def evaluate(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted'),
            "recall": recall_score(y_true, y_pred, average='weighted'),
            "f1_score": f1_score(y_true, y_pred, average='weighted')
        }

    @staticmethod
    def convert_to_sentiment_category(score):
        score = float(score)
        if 2. <= score <= 3.:
            return 'strongly positive'
        elif 1. <= score < 2.:
            return 'positive'
        elif 0. < score < 1.:
            return 'weakly positive'
        elif score == 0.:
            return 'neutral'
        elif -1. < score < 0.:
            return 'weakly negative'
        elif -2. < score <= -1.:
            return 'negative'
        elif -3. <= score <= -2.:
            return 'strongly negative'
        else:
            return 'unknown'