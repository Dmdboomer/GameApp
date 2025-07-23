from Model_Loader import evaluate_position
import torch

MODEL_1 = 'PytorchModel/chess_eval_model_5MParameters_25epochs.pth'
MODEL_2 = 'PytorchModel/chess_eval_model_5MParameters_3epochs.pth'
TEST_FILE = 'Data/test_data_5M.txt'

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

def closer_times(mod1, mod2):
    model_1_points = 0
    model_2_points = 0
    print("BEGIN Comparision TEST: ")
    with open(TEST_FILE, 'r') as f:
        for line in f:
            pos, eval = line.strip().split('|')
            eval = parse_evaluation(eval)
            mod1_eval = evaluate_position(pos, mod1)
            mod2_eval = evaluate_position(pos, mod2)
            error1 = abs(eval - mod1_eval)
            error2 = abs(eval - mod2_eval)
            if error1 < error2:  # Model 1 is closer
                model_1_points += 1
            elif error2 < error1:  # Model 2 is closer
                model_2_points += 1
    print("Model 1: %d points", model_1_points)
    print("Model 2: %d points", model_2_points) 

def times_really_close(mod1, mod2):
    model_1_points = 0
    model_2_points = 0
    print("BEGIN Absolute TEST: ")
    with open(TEST_FILE, 'r') as f:
        for line in f:
            pos, eval = line.strip().split('|')
            eval = parse_evaluation(eval)
            mod1_eval = evaluate_position(pos, mod1)
            mod2_eval = evaluate_position(pos, mod2)
            if abs(eval - mod1_eval) < 1:
                model_1_points+=1
            if abs(eval - mod2_eval) < 1:
                model_2_points +=1
    print("Model 1: %d points", model_1_points)
    print("Model 2: %d points", model_2_points) 
#Test:
if __name__ == "__main__":
    closer_times(MODEL_1, MODEL_2)  
    times_really_close(MODEL_1, MODEL_2)