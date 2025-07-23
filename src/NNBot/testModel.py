from Model_Loader import evaluate_position

fen_string = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
evaluation = evaluate_position(fen_string)
print(f"Predicted evaluation: {evaluation:.2f}")