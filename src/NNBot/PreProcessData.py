import numpy as np
import chess
import chess.pgn
import re
import random

MAX_POSITIONS = 100000

def filter_games_with_evals(input_file, output_file):
    games = []
    current_game = []
    game_has_eval = False
    counter = 0

    with open(input_file, 'r') as f:
        for line in f:
            stripped_line = line.rstrip()
            
            if stripped_line.startswith('[Event '):
                print(f"Processing game: {counter}")
                counter += 1
                if current_game and game_has_eval:
                    games.append("\n".join(current_game) + "\n\n")
                
                current_game = [stripped_line]
                game_has_eval = False
            else:
                current_game.append(stripped_line)
                if '[%eval' in stripped_line:
                    game_has_eval = True
    
    if current_game and game_has_eval:
        games.append("\n".join(current_game))
    
    with open(output_file, 'w') as f:
        f.write("\n".join(games))
    
    return output_file

def parse_and_save_positions_optimized(input_file, output_file, test_file, batch_size=100):
    eval_pattern = re.compile(r'%eval\s+([^\]\s]+)')
    batch_buffer = []
    test_batch_buffer = []
    index = 0
    
    with open(input_file, 'r', buffering=1048576) as pgn, \
         open(output_file, 'w', buffering=1048576) as out_f, \
         open(test_file, 'w', buffering=1048576) as test_f:

        while (game := chess.pgn.read_game(pgn)):
            board = game.board()
            for node in game.mainline():
                index += 1
                if node.comment and (match := eval_pattern.search(node.comment)):
                    position_data = f"{board.fen()}|{match.group(1)}\n"
                    
                    # Randomly select 1/1000 positions for test data
                    if random.random() < 0.001:
                        test_batch_buffer.append(position_data)
                    else:
                        batch_buffer.append(position_data)
                
                board.push(node.move)
                
                # Batch write
                if len(batch_buffer) >= batch_size:
                    out_f.writelines(batch_buffer)
                    batch_buffer.clear()
                
                if len(test_batch_buffer) >= batch_size:
                    test_f.writelines(test_batch_buffer)
                    test_batch_buffer.clear()
            
            # Periodic progress (per game)
            print(f"Processed {index} positions", end='\r')

            if index > MAX_POSITIONS:
                break
        
        # Flush final batches
        if batch_buffer:
            out_f.writelines(batch_buffer)
        if test_batch_buffer:
            test_f.writelines(test_batch_buffer)
    
if __name__ == "__main__":
    input_filename = "../Data/lichess_db_standard_rated_2014-07.txt"
    filtered_filename = "./Data/filtered_games.txt"
    parsed_file = "./Data/parsed_positions_100k.txt"
    test_file = "./Data/test_data_100k.txt"
    
    # Step 1: Filter games containing evaluations
    #filtered_file = filter_games_with_evals(input_filename, filtered_filename)
    #print(f"Filtered games saved to: {filtered_filename}")

    # Step 2: Parse the positions and evaluations
    parse_and_save_positions_optimized(filtered_filename, parsed_file, test_file, 100)