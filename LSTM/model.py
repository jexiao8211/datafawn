import torch
import torch.nn as nn

class FawnLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        """
        LSTM model for sequence labeling (binary classification per episode)
        
        Args:
            input_dim: Dimension of input vectors
            hidden_dim: Dimension of LSTM hidden state
            num_layers: Number of LSTM layers
        """
        super(FawnLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # for classification
        self.fc = nn.Linear(hidden_dim, 1)
        

        
    def forward(self, x):
        """
        Forward propagation
        """
        # x = (batch_size, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # (batch_size, seq_len, 1)
        logits = self.fc(lstm_out)
        # sigmoid to get probabilities, (batch_size, seq_len)
        probs = torch.sigmoid(logits).squeeze(-1)
        
        return logits, probs
    


    def predict(self, x, threshold=0.5):
        with torch.no_grad():
            probs = self.forward(x)
            predictions = (probs >= threshold).long()
        return predictions