import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader

# Piece mapping: {'p': 1, 'n': 2, ..., 'K': 12}
piece_to_idx = {char: idx for idx, char in enumerate('_pnbrqkPNBRQK', 1)}

def fen_to_features(fen):
    board, turn, *_ = fen.split()
    features = []
    
    # Parse board configuration (64 squares)
    for row in board.split('/'):
        for char in row:
            if char.isdigit():
                features.extend([0] * int(char))  # Empty squares
            else:
                features.append(piece_to_idx[char])
    
    # Add turn feature (1 for white, 0 for black)
    features.append(1 if turn == 'w' else 0)
    return torch.tensor(features, dtype=torch.float32)

def parse_evaluation(eval_str):
    if eval_str.startswith('#'):
        return 1000 - int(eval_str[1:])  # White winning
    elif eval_str.startswith('#-'):
        return -1000 + int(eval_str[2:])  # Black winning
    else:
        return float(eval_str)

class ChessEvalDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as f:
            index = 0
            for line in f:
                fen, eval_str = line.strip().split('|')
                features = fen_to_features(fen)
                if index % 1000 == 0:
                    print("Parsed: %d", index)
                index +=1
                eval_val = parse_evaluation(eval_str)
                self.data.append((features, torch.tensor([eval_val])))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class ChessEvalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(14, 4)  # 12 pieces + empty
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 + 1, 128),  # 64 squares * 4 emb dim + turn
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        board = x[:, :-1].long()  # First 64 features
        turn = x[:, -1:]  # Last feature
        emb = self.embedding(board).view(x.size(0), -1)
        return self.fc(torch.cat([emb, turn], dim=1))
    
# Initialize components
dataset = ChessEvalDataset('Data/parsed_positions.txt')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model = ChessEvalModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
for epoch in range(10):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
torch.save(model.state_dict(), 'chess_eval_model.pth')
print("Model saved to chess_eval_model.pth")

def evaluate_position(fen_str):
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        features = fen_to_features(fen_str).unsqueeze(0)
        prediction = model(features)
        return prediction.item()

# Example usage
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
print(f"Position evaluation: {evaluate_position(fen):.2f}")
