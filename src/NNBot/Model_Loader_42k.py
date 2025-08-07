import torch
from torch import nn
DEFAULT_MODEL = 'PytorchModel/chess_eval_model_100kParameters_25epochs.pth'

# Define model architecture (MUST match training)
class ChessEvalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(14, 4)  # 14 possible values (0-13)
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 + 5, 128),  # 64 squares * 4 emb dim + 5 features
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        board = x[:, :64].long()
        other = x[:, 64:]
        emb = self.embedding(board).view(x.size(0), -1)
        return self.fc(torch.cat([emb, other], dim=1))

# CORRECTED feature extraction (matches training)
piece_to_idx = {char: idx for idx, char in enumerate('_pnbrqkPNBRQK', 1)}
piece_to_idx['.'] = 0  # Critical for empty squares

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
def evaluate_position_42k(fen_str, curr_model = DEFAULT_MODEL):
    model = ChessEvalModel()
    model.load_state_dict(torch.load(curr_model))
    model.eval()
    with torch.no_grad():
        features = fen_to_features(fen_str).unsqueeze(0)
        return model(features).item()


# Example usage
if __name__ == "__main__":
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    print(f"Evaluation: {evaluate_position(fen, DEFAULT_MODEL):.2f}")