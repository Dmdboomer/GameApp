from Model_Loader_42k import evaluate_position_42k
from Model_Loader_360k import evaluate_position_360k
from Model_Loader_360k_v2 import evaluate_position_360k_v2
import chess
import torch
import math
# Helper functions with chess logic

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

def get_best_move(fen, model, eval):
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
    i = 0
    for move in legal_moves:
        new_board = board.copy()
        new_board.push(move)
        # returns the evaluation of a FEN
        score = eval(new_board.fen(), model)
        
        # Always look for the position with the highest evaluation score.
        if score > best_score:
            best_score = score
            best_move = move
        i +=1

    return best_move


# Main Scoring Functions

def closer_times(models):
    modelScore = [0] * len(models)
    print("BEGIN Comparision TEST: ")
    with open(TEST_FILE, 'r') as f:
        for line in f:
            pos, eval = line.strip().split('|')
            eval = parse_evaluation(eval)
            bestScore = 10**7
            currBest = -1
            for i in range(len(models)):
                evaluate_position = models[i][1]
                currScore = abs(eval - evaluate_position(pos, models[i][0]))
                if currScore < bestScore:
                    if currBest >= 0:
                        modelScore[currBest] -= 1
                    currBest = i
                    bestScore = currScore
                    modelScore[i] +=1
                    

    for i in range(len(modelScore)):
        print("Model", (i+1) , ":", modelScore[i], "points")
    

def times_really_close(models):
    modelPoints = [0] * (len(models))
    print("BEGIN Absolute TEST: ")
    with open(TEST_FILE, 'r') as f:
        for line in f:
            pos, eval = line.strip().split('|')
            eval = parse_evaluation(eval)
            for i in range(len(modelPoints)):
                mod_eval = models[i][1](pos, models[i][0])
                if abs(eval - mod_eval) < 1:
                    modelPoints[i] += 1

    for i in range(len(modelPoints)):
        print("Model", (i+1) , ":", modelPoints[i], "points")

def compareMoves(fen, models):
    for i in range(len(models)):
        bestMove = get_best_move(fen, models[i][0], models[i][1])
        print("Model: ", (i+1), ": ", bestMove)

#Test:
if __name__ == "__main__":
    MODEL_1 = ['./PytorchModels/42kParameter_100kPosition_200E_1in10.pth', evaluate_position_42k]
    MODEL_2 = ['./PytorchModels/360kParameter_5MPosition_1E_1in10.pth', evaluate_position_360k]
    MODEL_3 = ['./PytorchModels/360kParameter_1MPosition_100E_1in10.pth', evaluate_position_360k]
    MODEL_4 = ['./PytorchModels/42kParameter_5MPosition_XE_1in10.pth', evaluate_position_42k]
    MODEL_5 = ['./PytorchModels/360kParameter_5MPosition_XE_1in10.pth', evaluate_position_360k]
    MODEL_6 = ['./PytorchModels/360kParameter_5MPosition_XE_v2.pth', evaluate_position_360k_v2]
    ALL_MODELS = [MODEL_1, MODEL_2, MODEL_3, MODEL_4, MODEL_5, MODEL_6]

    
    ALL_NAMES = [''] * len(ALL_MODELS)
    ALL_EVALFUNCTIONS = [''] * len(ALL_MODELS)

    for i in range(len(ALL_MODELS)):
        ALL_NAMES[i] = ALL_MODELS[i][0]
        ALL_EVALFUNCTIONS[i] = ALL_MODELS[i][1]


    TEST_FILE = 'Data/clean_test_5M.txt'
    closer_times(ALL_MODELS) 
    times_really_close(ALL_MODELS)
    test_fen = "r1bq1rk1/ppppbppp/2n1p3/3nP3/3P2Q1/2PB4/PP1B1PPP/RN2K1NR w KQ - 5 8"
    #compareMoves(test_fen, ALL_MODELS)
