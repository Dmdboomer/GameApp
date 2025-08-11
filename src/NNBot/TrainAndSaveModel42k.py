import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader

def fen_to_features(fen):
    tokens = fen.split()
    board = tokens[0]
    turn = tokens[1]
    castling = tokens[2] if len(tokens) > 2 else '-'  # Handle missing castling
    
    features = []
    
    # Parse board configuration
    for row in board.split('/'):
        for char in row:
            if char.isdigit():
                features.extend([0] * int(char))  # Empty squares
            elif char in piece_to_idx:
                features.append(piece_to_idx[char])
            else:
                features.append(0)  # Fallback for unknown
    
    # Add turn feature (1 for white, 0 for black)
    features.append(1 if turn == 'w' else 0)
    
    # Add castling features (KQkq order)
    castling_map = {'K': 0, 'Q': 1, 'k': 2, 'q': 3}
    castling_features = [0, 0, 0, 0]
    if castling != '-':
        for char in castling:
            if char in castling_map:
                castling_features[castling_map[char]] = 1
    features.extend(castling_features)
    
    return torch.tensor(features, dtype=torch.float32)

def parse_evaluation(eval_str):
    # Handle draws
    if eval_str == '1/2-1/2':
        return 0.0
    
    # Handle mate evaluations
    if eval_str.startswith('#'):
        if eval_str.startswith('#-'):
            return -1000 + int(eval_str[2:])
        return 1000 - int(eval_str[1:])
    
    # Fallback to float conversion
    try:
        return float(eval_str)
    except:
        return 0.0  # Default for unparseable

class ChessEvalDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as f:
            index = 0
            for line in f:
                fen, eval_str = line.strip().split('|')
                features = fen_to_features(fen)
                if index % 10000 == 0:
                    print(f"Parsed: {index}")
                index += 1
                eval_val = parse_evaluation(eval_str)
                self.data.append((features, torch.tensor([eval_val])))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class ChessEvalModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 14 embeddings: 13 pieces + empty (0 index)
        self.embedding = nn.Embedding(14, 4)  
        
        # New dimensions: 64 squares * 4 + 5 features (turn + 4 castling)
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 + 5, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        board = x[:, :64].long()       # First 64 features (board)
        other_features = x[:, 64:]     # Last 5 features (turn + castling)
        
        # Embed and flatten board
        emb = self.embedding(board).view(x.size(0), -1)
        
        # Combine with other features
        combined = torch.cat([emb, other_features], dim=1)
        
        return self.fc(combined)

def evaluate_position(fen_str):
    model.load_state_dict(torch.load(PATH_TO_MODEL))
    model.eval()
    with torch.no_grad():
        features = fen_to_features(fen_str).unsqueeze(0)
        prediction = model(features)
        return prediction.item()

if __name__ == '__main__':  # Wrap main code
    # Piece mapping with blank character included
    piece_to_idx = {char: idx for idx, char in enumerate('_pnbrqkPNBRQK', 1)}
    piece_to_idx['.'] = 0  # Explicit mapping for empty squares
    PATH_TO_DATA = 'Data/clean_position_5M.txt'
    PATH_TO_TEST = './Data/clean_test_5M.txt'
    PATH_TO_MODEL = 'PytorchModels/42kParameter_5MPosition_XE_1in10.pth'
    EPOCH_AMOUNT = 3000
    EARLY_STOP_PATIENCE = 10  # Stop if no improvement for 2 epochs

    # Create datasets
    train_dataset = ChessEvalDataset(PATH_TO_DATA)
    test_dataset = ChessEvalDataset(PATH_TO_TEST)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = ChessEvalModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Early stopping initialization
    best_val_accuracy = 0.0  # Track best accuracy instead of loss
    epochs_no_improve = 0

    for epoch in range(EPOCH_AMOUNT):
        # Training phase
        model.train()
        train_loss_total = 0
        index = 0
        for inputs, targets in train_dataloader:
            if index % 10000 == 0:
                print(f"Trained on batch: {index}")
            index += 1
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
        
        # Validation phase
        model.eval()
        val_loss_total = 0
        val_correct = 0  # Count of predictions within 0.15 of target
        val_total = 0    # Total validation samples
        
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                outputs = model(inputs)
                val_loss_total += criterion(outputs, targets).item()
                
                # Calculate accuracy within tolerance
                diff = torch.abs(outputs - targets)
                correct = torch.sum(diff < 1).item()
                val_correct += correct
                val_total += targets.size(0)
        
        avg_val_loss = val_loss_total / len(test_dataloader)
        val_accuracy = val_correct / val_total
        
        print(f'Epoch {epoch+1} | Train Loss: {train_loss_total/len(train_dataloader):.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}')

        # Early stopping check based on accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
            torch.save(model.state_dict(), PATH_TO_MODEL)
            print(f"Validation accuracy improved to {best_val_accuracy:.4f}! Saved model to {PATH_TO_MODEL}")
        else:
            epochs_no_improve += 1
            print(f"No validation improvement ({epochs_no_improve}/{EARLY_STOP_PATIENCE})")
            
        # Stop if no improvement for EARLY_STOP_PATIENCE epochs
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stopping after {epoch+1} epochs")
            break

    torch.save(model.state_dict(), PATH_TO_MODEL)
    print("Model saved to " + PATH_TO_MODEL)
    # Example usage with castling
    fen = "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    print(f"Position evaluation: {evaluate_position(fen):.2f}")
