import numpy as np
import chess
import chess.pgn
import re
import random
import io
import zstandard as zstd 

MAX_POSITIONS = 10000000

def stream_decompressed_games(input_file):
    """Generator to stream decompressed games line by line"""
    cctx = zstd.ZstdDecompressor()
    with open(input_file, 'rb') as fh:
        reader = cctx.stream_reader(fh)
        text_reader = io.TextIOWrapper(reader, encoding='utf-8')
        for line in text_reader:
            yield line

def filter_games_with_evals(input_file, output_file):
    current_game = []
    game_has_eval = False
    counter = 0

    with open(output_file, 'w', encoding='utf-8') as out_f:
        line_generator = stream_decompressed_games(input_file)
        
        for line in line_generator:
            stripped_line = line.rstrip()
            
            if stripped_line.startswith('[Event '):
                # Process previous game before starting new one
                if current_game:
                    if game_has_eval:
                        out_f.write("\n".join(current_game))
                        out_f.write("\n\n")  # Consistent game separator
                    current_game = []
                    game_has_eval = False
                
                # Update counter after processing previous game
                counter += 1
                if counter % 100000 == 0:
                    print(f"Processing game: {counter}")
            
            # Check for eval in non-event lines
            if '[%eval' in stripped_line:
                game_has_eval = True
                
            current_game.append(stripped_line)
        
        # Process final game after loop completes
        if current_game and game_has_eval:
            out_f.write("\n".join(current_game))
            out_f.write("\n\n")
    
    return output_file

def parse_and_save_positions_optimized(input_file, output_file, test_file, batch_size=100):
    eval_pattern = re.compile(r'%eval\s+([^\]\s]+)')
    batch_buffer = []
    test_batch_buffer = []
    index = 0
    match_count = 0  # Tracks matching positions for sampling[1,4](@ref)
    
    with open(input_file, 'r', buffering=1048576) as pgn, \
         open(output_file, 'w', buffering=1048576) as out_f, \
         open(test_file, 'w', buffering=1048576) as test_f:

        while (game := chess.pgn.read_game(pgn)):
            board = game.board()
            for node in game.mainline():
                index += 1
            
                board.push(node.move)  # Maintain board state progression

                # Process only matching comments
                if node.comment and (match := eval_pattern.search(node.comment)):
                    match_count += 1
                    
                    # Take only 1/10 matching positions
                    if match_count % 10 == 0:  # Deterministic sampling[1,4](@ref)
                        position_data = f"{board.fen()}|{match.group(1)}\n"
                        
                        # Randomly select 1/1000 for test data (from sampled positions)
                        if random.random() < 0.001:
                            test_batch_buffer.append(position_data)
                        else:
                            batch_buffer.append(position_data)
                                
                # Batch writing (reduced frequency due to sampling)
                if len(batch_buffer) >= batch_size:
                    out_f.writelines(batch_buffer)
                    batch_buffer.clear()
                if len(test_batch_buffer) >= batch_size:
                    test_f.writelines(test_batch_buffer)
                    test_batch_buffer.clear()
            if index % 100 == 0:
                print(f"Processed {index} positions", end='\r')
            if index > MAX_POSITIONS:
                break
        
        # Final buffer flush
        if batch_buffer:
            out_f.writelines(batch_buffer)
        if test_batch_buffer:
            test_f.writelines(test_batch_buffer)
    
if __name__ == "__main__":
    # Update to your compressed input file
    input_filename = "/Users/yc/Downloads/lichess_db_standard_rated_2025-07.pgn.zst"
    filtered_filename = "./Data/filtered_games.txt"
    parsed_file = "./Data/filtered_positions_1M.txt"
    test_file = "./Data/test_positions_1M.txt"
    
    # Step 1: Filter games containing evaluations
    #filtered_file = filter_games_with_evals(input_filename, filtered_filename)
    #print(f"Filtered games saved to: {filtered_filename}")

    # Step 2: Parse the positions and evaluations
    parse_and_save_positions_optimized(filtered_filename, parsed_file, test_file, 100)
    
    # Step 3: Remove duplicates

