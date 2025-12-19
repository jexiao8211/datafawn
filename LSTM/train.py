from model import FawnLSTM
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim


def train(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"Now training LSTM on {device}")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits, _ = model(batch_x)
            
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                logits, _ = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
    
    return train_losses, val_losses


train_loader, val_loader, test_loader = get_dataloader()
model = FawnLSTM()

train_losses, val_losses = train(model=model, train_loader=train_loader, val_loader=val_loader)