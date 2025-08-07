import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import DataLoader

import torch
piece_to_idx = {char: idx for idx, char in enumerate('_pnbrqkPNBRQK', 1)}
piece_to_idx['.'] = 0

def fen_to_features(fen):
    tokens = fen.split()
    board = tokens[0]
    turn = tokens[1]
    castling = tokens[2] if len(tokens) > 2 else '-'
    
    features = []
    for row in board.split('/'):
        for char in row:
            if char.isdigit():
                features.extend([0] * int(char))
            elif char in piece_to_idx:
                features.append(piece_to_idx[char])
            else:
                features.append(0)
                
    features.append(1 if turn == 'w' else 0)
    
    castling_map = {'K': 0, 'Q': 1, 'k': 2, 'q': 3}
    castling_features = [0, 0, 0, 0]
    if castling != '-':
        for char in castling:
            if char in castling_map:
                castling_features[castling_map[char]] = 1
    features.extend(castling_features)
    
    return torch.tensor(features, dtype=torch.float32)

def parse_evaluation(eval_str):
    if eval_str == '1/2-1/2':
        return 0.0
    if eval_str.startswith('#'):
        if eval_str.startswith('#-'):
            return -1000 + int(eval_str[2:])
        return 1000 - int(eval_str[1:])
    try:
        return float(eval_str)
    except:
        return 0.0

class ChessEvalDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as f:
            for idx, line in enumerate(f):
                fen, eval_str = line.strip().split('|')
                features = fen_to_features(fen)
                if idx % 10000 == 0:
                    print(f"Parsed: {idx}")
                eval_val = parse_evaluation(eval_str)
                self.data.append((features, torch.tensor([eval_val])))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class SmallChessEvalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(14, 32)  # Reduced from 64 to 32
        
        # Simplified residual blocks with channel reduction
        self.residual_proj1 = nn.Conv2d(32, 64, kernel_size=1, bias=False)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        
        self.residual_proj2 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        
        # Transition with pooling to reduce features
        self.transition = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Pool to 1x1 features
        )
        
        # Streamlined fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256 + 5, 128),  # Input features: 256 + 5 (metadata)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.relu = nn.ReLU()
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {total_params:,}")

    def forward(self, x):
        board = x[:, :64].long()
        other_features = x[:, 64:]
        
        emb = self.embedding(board)
        #emb = emb.view(-1, 8, 8, 32).permute(0, 3, 1, 2)  # [B, 32, 8, 8]
        emb = emb.reshape(-1, 8, 8, 32).permute(0, 3, 1, 2).contiguous()

        # Residual Block 1
        residual1 = self.residual_proj1(emb)
        conv_out = self.conv_block1(emb)
        conv_out = self.relu(conv_out + residual1)
        
        # Residual Block 2
        residual2 = self.residual_proj2(conv_out)
        conv_out = self.conv_block2(conv_out)
        conv_out = self.relu(conv_out + residual2)
        
        # Transition and flatten
        conv_out = self.transition(conv_out)

        #conv_out = conv_out.view(conv_out.size(0), -1)
        conv_out = conv_out.reshape(conv_out.size(0), -1).contiguous()

        
        # Combine with metadata features
        combined = torch.cat([conv_out, other_features], dim=1)
        return self.fc(combined)
    
def evaluate_position(fen_str):
    model.load_state_dict(torch.load(PATH_TO_MODEL))
    model.eval()
    with torch.no_grad():
        features = fen_to_features(fen_str).unsqueeze(0)
        prediction = model(features)
        return prediction.item()

if __name__ == '__main__':
    piece_to_idx = {char: idx for idx, char in enumerate('_pnbrqkPNBRQK', 1)}
    piece_to_idx['.'] = 0
    PATH_TO_DATA = './Data/filtered_positions_1M.txt'
    PATH_TO_MODEL = 'PytorchModels/360kParameter_1MPosition_100E_1in10.pth'
    EPOCH_AMOUNT = 100

    dataset = ChessEvalDataset(PATH_TO_DATA)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = SmallChessEvalModel()  # Use the new smaller model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(EPOCH_AMOUNT):
        epoch_loss = 0.0
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if idx % 1000 == 0:
                print(f"Batch {idx}, Loss: {loss.item():.4f}")
        print(f'Epoch {epoch+1}, Average Loss: {epoch_loss/len(dataloader):.4f}')

    torch.save(model.state_dict(), PATH_TO_MODEL)
    print(f"Model saved to {PATH_TO_MODEL}")
    
    fen = "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    print(f"Position evaluation: {evaluate_position(fen):.2f}")