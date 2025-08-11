import torch
from torch import nn
import math


DEFAULT_MODEL = '/Users/yc/Documents/GameEngine/PytorchModels/360kParameter_5MPosition_XE_v2.pth'

def winprob_to_centipawn(win_prob):
    """
    Convert win probability in [-1, 1] range to centipawn value
    Formula: cp = -4 * log10((1 - p) / p) where p = (win_prob + 1)/2
    """
    # Handle extreme cases first
    if win_prob >= 0.999:
        return 1000.0  # White mate
    elif win_prob <= -0.999:
        return -1000.0  # Black mate
    elif abs(win_prob) < 0.001:
        return 0.0  # Draw
    
    # Convert to [0, 1] range
    p = (win_prob + 1) / 2.0
    
    # Apply sigmoid inverse
    ratio = (1 - p) / p
    # Avoid log(0)
    if ratio < 1e-12:
        return 1000.0 if win_prob > 0 else -1000.0
    
    cp = -4 * math.log10(ratio)
    return cp

class ChessEvalModel360k(nn.Module):
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

# Feature extraction mapping
piece_to_idx = {char: idx for idx, char in enumerate('_pnbrqkPNBRQK', 1)}
piece_to_idx['.'] = 0

def fen_to_features(fen):
    """Revised to return SEPARATE board and metadata tensors"""
    tokens = fen.split()
    board_str = tokens[0]
    turn = tokens[1]
    castling = tokens[2] if len(tokens) > 2 else '-'
    
    # Parse board
    board_features = []
    for row in board_str.split('/'):
        for char in row:
            if char.isdigit():
                board_features.extend([0] * int(char))
            else:
                board_features.append(piece_to_idx.get(char, 0))  # Fallback to 0
    
    # Metadata features
    metadata_features = [
        1 if turn == 'w' else 0,  # Turn feature
        *[1 if flag in castling else 0 for flag in ['K', 'Q', 'k', 'q']]  # Castling flags
    ]
    
    return (
        torch.tensor(board_features, dtype=torch.long),  # Board tensor (int64)
        torch.tensor(metadata_features, dtype=torch.float32)  # Metadata tensor
    )

def evaluate_position_360k_v2(fen_str, curr_model=DEFAULT_MODEL):
    model = ChessEvalModel360k()
    
    # Load model with CPU device mapping
    device = torch.device('cpu')
    model.load_state_dict(
        torch.load(curr_model, map_location=device)
    )
    model.eval()
    
    with torch.no_grad():
        board_tensor, metadata_tensor = fen_to_features(fen_str)
        
        # Reshape and add batch dimension
        board_input = board_tensor.reshape(1, 8, 8)  # [1, 8, 8]
        metadata_input = metadata_tensor.unsqueeze(0)  # [1, 5]
        
        # Pass both required arguments
        return winprob_to_centipawn(model(board_input, metadata_input).item())

# Test with sample position
if __name__ == "__main__":
    test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    print(evaluate_position_360k_v2(test_fen, DEFAULT_MODEL))