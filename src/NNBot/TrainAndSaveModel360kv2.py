import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import copy
import re

# Piece mapping dictionary
piece_to_idx = {char: idx for idx, char in enumerate('_pnbrqkPNBRQK', 1)}
piece_to_idx['.'] = 0

def fen_to_features(fen):
    tokens = fen.split()
    board_str = tokens[0]
    turn = tokens[1]
    castling = tokens[2] if len(tokens) > 2 else '-'
    
    # Parse board into 8x8 grid
    board_1d = []
    for row in board_str.split('/'):
        for char in row:
            if char.isdigit():
                board_1d.extend([0] * int(char))
            else:
                board_1d.append(piece_to_idx[char])
    
    # Convert to 8x8 tensor (long)
    board_tensor = torch.tensor(board_1d, dtype=torch.long).view(8, 8)
    
    # Metadata features
    turn_feature = 1.0 if turn == 'w' else 0.0
    
    castling_map = {'K': 0, 'Q': 1, 'k': 2, 'q': 3}
    castling_features = [0.0, 0.0, 0.0, 0.0]
    if castling != '-':
        for char in castling:
            if char in castling_map:
                castling_features[castling_map[char]] = 1.0
    
    metadata = torch.tensor([turn_feature] + castling_features, dtype=torch.float32)
    return board_tensor, metadata

def parse_evaluation(eval_str, turn):
    """Convert evaluation to win probability in [-1, 1] range"""
    # Handle draws
    if eval_str == '1/2-1/2':
        return 0.0
    
    # Handle mate scores
    if eval_str.startswith('#'):
        if eval_str.startswith('#-'):
            return -1.0 if turn == 'w' else 1.0
        else:
            return 1.0 if turn == 'w' else -1.0
    
    # Centipawn conversion to win probability
    try:
        cp_val = float(eval_str)
    except ValueError:
        return 0.0
    
    # Adjust for current player perspective
    if turn == 'b':
        cp_val = -cp_val
    
    # Sigmoid conversion: 2/(1 + 10^(-cp/4)) - 1 → [-1, 1]
    win_prob = 1.0 / (1 + 10 ** (-cp_val / 4))
    return 2 * win_prob - 1

class ChessEvalDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as f:
            for idx, line in enumerate(f):
                fen, eval_str = line.strip().split('|', 1)
                board, metadata = fen_to_features(fen)
                turn = fen.split()[1]  # Extract turn from FEN
                eval_val = parse_evaluation(eval_str, turn)
                self.data.append((board, metadata, torch.tensor([eval_val])))
                
                if idx % 10000 == 0:
                    print(f"Parsed: {idx}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        board, metadata, eval_val = self.data[idx]
        return (board, metadata), eval_val

class SmallChessEvalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(14, 32)  # 13 pieces + empty
        
        # Residual convolution blocks
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
        
        # Feature compression
        self.transition = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256 + 5, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.relu = nn.ReLU()
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {total_params:,}")

    def forward(self, board, metadata):
        # Embedding: [B, 8, 8] → [B, 8, 8, 32] → [B, 32, 8, 8]
        emb = self.embedding(board)
        emb = emb.permute(0, 3, 1, 2).contiguous()
        
        # Residual block 1
        residual1 = self.residual_proj1(emb)
        conv_out = self.conv_block1(emb)
        conv_out = self.relu(conv_out + residual1)
        
        # Residual block 2
        residual2 = self.residual_proj2(conv_out)
        conv_out = self.conv_block2(conv_out)
        conv_out = self.relu(conv_out + residual2)
        
        # Compress features
        conv_out = self.transition(conv_out)
        conv_out = conv_out.view(conv_out.size(0), -1)
        
        # Combine with metadata
        combined = torch.cat([conv_out, metadata], dim=1)
        return self.fc(combined)

class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)

if __name__ == '__main__':
    PATH_TO_DATA = './Data/clean_position_5M.txt'
    PATH_TO_MODEL = 'PytorchModels/360kParameter_5MPosition_XE_v2.pth'
    PATH_TO_TEST = './Data/clean_test_5M.txt'
    EPOCH_AMOUNT = 30
    EARLY_STOP_PATIENCE = 3

    # Initialize datasets
    train_dataset = ChessEvalDataset(PATH_TO_DATA)
    test_dataset = ChessEvalDataset(PATH_TO_TEST)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model setup
    model = SmallChessEvalModel()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    # Training loop
    for epoch in range(EPOCH_AMOUNT):
        model.train()
        epoch_loss = 0.0
        for idx, (inputs, targets) in enumerate(train_loader):
            board_batch = inputs[0].to(device)
            meta_batch = inputs[1].to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(board_batch, meta_batch)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if idx % 1000 == 0:
                print(f"Batch {idx}, Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_points = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                board_batch = inputs[0].to(device)
                meta_batch = inputs[1].to(device)
                targets = targets.to(device)
                
                outputs = model(board_batch, meta_batch)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * targets.size(0)
                
                abs_errors = torch.abs(outputs - targets).squeeze()
                correct_points += (abs_errors <= 0.05).sum().item()
                total_samples += targets.size(0)
        
        val_loss /= total_samples
        accuracy = correct_points / total_samples
        print(f'Epoch {epoch+1}, Train Loss: {epoch_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss:.4f}, Accuracy (within 0.3): {accuracy:.4f}')
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Save final model
    early_stopping.load_best_model(model)
    torch.save(model.state_dict(), PATH_TO_MODEL)
    print(f"Model saved to {PATH_TO_MODEL}")