from Model_Loader import evaluate_position

MODEL_1 = 'PytorchModel/chess_eval_model_1MParameters_25epochs.pth'
MODEL_2 = 'PytorchModel/chess_eval_model_100kParameters_1epochs.pth'
TEST_FILE = 'Data/test_data.txt'

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
            if abs(eval) - abs(mod1_eval) > abs(eval) - abs(mod2_eval):
                model_2_points+=1
            else:
                model_1_points +=1
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
            if abs(eval) - abs(mod1_eval) < 0.5:
                model_2_points+=1
            if abs(eval) - abs(mod2_eval) < 0.5:
                model_1_points +=1
    print("Model 1: %d points", model_1_points)
    print("Model 2: %d points", model_2_points) 
#Test:
if __name__ == "__main__":
    closer_times(MODEL_1, MODEL_2)  
    times_really_close(MODEL_1, MODEL_2)