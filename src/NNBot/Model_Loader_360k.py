import torch
from torch import nn
DEFAULT_MODEL = '/Users/yc/Documents/GameEngine/PytorchModels/360kParameter_100kposition_5E_1in10.pth'

class ChessEvalModel360k(nn.Module):
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

    def forward(self, x):
        board = x[:, :64].long()
        other_features = x[:, 64:]
        
        emb = self.embedding(board)
        emb = emb.view(-1, 8, 8, 32).permute(0, 3, 1, 2)  # [B, 32, 8, 8]
        
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
        conv_out = conv_out.view(conv_out.size(0), -1)
        
        # Combine with metadata features
        combined = torch.cat([conv_out, other_features], dim=1)
        return self.fc(combined)

# CORRECTED feature extraction (matches training)
piece_to_idx = {char: idx for idx, char in enumerate('_pnbrqkPNBRQK', 1)}
piece_to_idx['.'] = 0

def fen_to_features(fen):
    tokens = fen.split()
    board = tokens[0]
    turn = tokens[1]
    castling = tokens[2] if len(tokens) > 2 else '-'

    features = []
    # Parse board
    for row in board.split('/'):
        for char in row:
            if char.isdigit():
                features.extend([0] * int(char))
            elif char in piece_to_idx:
                features.append(piece_to_idx[char])
            else:  # Fallback for unknowns
                features.append(0)
    
    # Add turn and castling features
    features.append(1 if turn == 'w' else 0)
    castling_flags = ['K','Q','k','q']
    features.extend([1 if c in castling else 0 for c in castling_flags])
    
    return torch.tensor(features, dtype=torch.float32)

# Inference function
def evaluate_position_360k(fen_str, curr_model = DEFAULT_MODEL):
    model = ChessEvalModel360k()
    model.load_state_dict(torch.load(curr_model))
    model.eval()
    with torch.no_grad():
        features = fen_to_features(fen_str).unsqueeze(0)
        return model(features).item()


# Example usage
if __name__ == "__main__":
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
