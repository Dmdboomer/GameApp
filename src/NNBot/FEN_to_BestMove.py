from Model_Loader import evaluate_position
import chess
import sys
MODEL_2 = 'PytorchModel/chess_eval_model_1MParameters_25epochs.pth'
MODEL_1 = 'PytorchModel/chess_eval_model_100kParameters_1epochs.pth'

def generate_next_fens(fen):
    """
    Generates all possible FENs resulting from legal moves in a given position.
    
    Args:
        fen (str): Input FEN string representing a chess position
       
    Returns:
        list: FEN strings of all positions after legal moves
    """
    try:
        board = chess.Board(fen)
        next_fens = []
        moves = []
        
        for move in board.legal_moves:
            # Create a copy of the board to avoid modifying the original
            new_board = board.copy()
            new_board.push(move)
            next_fens.append(new_board.fen())
            
        return next_fens
    
    except ValueError:
        raise ValueError("Invalid FEN string provided")


def get_best_move(fen, which_model = MODEL_1):
    """
    Finds the best move in the current position using the model for position evaluation.
    The model's evaluation inherently considers the active player (whose turn it is) 
    because the FEN string includes this information and the model was trained on FENs.
    """
    try:
        board = chess.Board(fen)
    except ValueError as e:
        raise ValueError(f"Invalid FEN string: {e}")

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None  # No legal moves available

    best_move = None
    best_score = float('-inf')  # Always seek the highest score, regardless of active color

    for move in legal_moves:
        new_board = board.copy()
        new_board.push(move)
        # returns the evaluation of a FEN
        score = evaluate_position(new_board.fen(), which_model)
        
        # Always look for the position with the highest evaluation score.
        if score > best_score:
            best_score = score
            best_move = move

    return best_move

# Example usage:
if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            fen = " ".join(sys.argv[1:])
            best_move_1 = get_best_move(fen, MODEL_1)
            best_move_2 = get_best_move(fen, MODEL_2)
            print(best_move_1.uci() if best_move_1 else "null")
            print(best_move_2.uci() if best_move_2 else "null")  
        else:
            print("Usage: python chess_ai.py <FEN>")
            sys.exit(1)
    except Exception as e:  # Catch-all for debugging
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)