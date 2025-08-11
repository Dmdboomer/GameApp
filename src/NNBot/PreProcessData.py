import numpy as np
import chess
import chess.pgn
import re
import random
import io
import zstandard as zstd 
import hashlib

MAX_POSITIONS = 50000000

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
                if counter % 10000 == 0:
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
    game_index = 0
    benchmarks = 0
    match_count = 0  # Tracks matching positions for sampling
    
    with open(input_file, 'r', buffering=1048576) as pgn, \
         open(output_file, 'w', buffering=1048576) as out_f, \
         open(test_file, 'w', buffering=1048576) as test_f:
         

        while (game := chess.pgn.read_game(pgn)):
            game_index +=1
            if game_index % 10000 == 0:
                print("Finished parsing game: ", game_index)
            board = game.board()
            for node in game.mainline():
                index += 1
            
                board.push(node.move)  # Maintain board state progression

                # Process only matching comments
                if node.comment and (match := eval_pattern.search(node.comment)):
                    match_count += 1
                    
                    # Take only 1/5 matching positions (or all for full then filter later)
                    if match_count % 5 == 0:  # Deterministic sampling
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
            if index > benchmarks:
                print(f"Processed {benchmarks} positions", end='\r')
                benchmarks += 10000
            #if index > 100000:
                #break
        
        # Final buffer flush
        if batch_buffer:
            out_f.writelines(batch_buffer)
        if test_batch_buffer:
            test_f.writelines(test_batch_buffer)


def count_lines_iterative(filepath):
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for _ in f: 
            count += 1
            if count % 1000000 == 0: 
                print("Line number: ", count)

def dedup_two_files(file1, file2, output_file):
    seen = set()
    index = 0
    with open(output_file, 'w') as out:
        for file in [file1, file2]:
            # Read as text, handle encoding issues
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:  
                for line in f:
                    line = line.rstrip('\n')  # Normalize line endings
                    if not line:
                        continue
                    index += 1
                    if index % 100000 == 0:
                        print("Processed:", index, "lines")
                    line_hash = hashlib.md5(line.encode()).hexdigest()
                    if line_hash not in seen:
                        seen.add(line_hash)
                        out.write(line + '\n')
                    # Optional flushing
                    if len(seen) % 1_000_000 == 0:  
                        out.flush()

def check_unique(file_path):
    """Check for duplicate lines in a file while tracking progress."""
    seen = set()
    try:
        with open(file_path, 'r') as f:  # Fix 1: Use read mode ('r')
            for index, line in enumerate(f, 1):
                line = line.rstrip('\n')  # Fix 2: Normalize line endings
                if line in seen:
                    print(f"DUPLICATE FOUND at line {index}: '{line}'")
                    return False
                seen.add(line)
                
                # Fix 3: Progress tracking at intervals
                if index % 10000 == 0:  # Reduced frequency for performance
                    print(f"Processed {index:,} lines...")
        print("All lines are unique!")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")

def mf(original_file_path: str, test_file_path: str) -> None:
    """
    Extracts 1/1000 lines from a large randomized text file into a test file,
    removes them from the original file, and overwrites the original.
    
    Args:
        original_file_path: Path to the 6GB source text file.
        test_file_path: Path to save the sampled lines (test file).
    """
    # Temporary files for safe write-and-replace
    temp_original_path = original_file_path + ".tmp"
    temp_test_path = test_file_path + ".tmp"
    
    try:
        with (
            open(original_file_path, 'r', encoding='utf-8') as src,
            open(temp_original_path, 'w', encoding='utf-8') as dest_original,
            open(temp_test_path, 'w', encoding='utf-8') as dest_test
        ):
            line_count = 0
            for line in src:
                line_count += 1
                # Write every 1000th line to the test file
                if line_count % 10000 == 0:
                    dest_test.write(line)
                else:  # Keep others in the original file
                    dest_original.write(line)
            if line_count % 1000 == 0:
                print("Line: ", line_count)
        # Replace original files with processed versions
        import os
        os.replace(temp_original_path, original_file_path)
        os.replace(temp_test_path, test_file_path)
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e.filename}") from e
    except Exception as e:
        # Clean up temporary files on failure
        import shutil
        shutil.rmtree(temp_original_path, ignore_errors=True)
        shutil.rmtree(temp_test_path, ignore_errors=True)
        raise RuntimeError(f"Operation failed: {str(e)}") from e

def smallerData(file, dest, amount):
    with (open(file, 'r') as src,
          open(dest, 'w') as dest
    ):
        index = 0
        for line in src:
            index +=1
            if index > amount:
                break
            dest.write(line)
            if index % 10000 == 0:
                print("Position: ", index)

if __name__ == "__main__":
    # Update to your compressed input file
    input_filename = "/Users/yc/Downloads/lichess_db_standard_rated_2025-07.pgn.zst"
    filtered_filename = "./Data/filtered_games.txt"
    parsed_file = "./Data/filtered_positions_full.txt"
    test_file = "./Data/test_positions_full.txt"
    clean_positions = "./Data/clean_position_full.txt"
    clean_test = "./Data/clean_test_full.txt"
    
    # Step 1: Filter games containing evaluations
    #filtered_file = filter_games_with_evals(input_filename, filtered_filename)
    #print(f"Filtered games saved to: {filtered_filename}")

    # Step 2: Parse the positions and evaluations
    #parse_and_save_positions_optimized(filtered_filename, parsed_file, test_file, 100)
    
    # Step 3: Remove duplicates, generate test_file
    #dedup_two_files(parsed_file, test_file, clean_positions)
    #mf(clean_positions, clean_test)

    #smallerData(clean_positions, './Data/clean_position_5M', 5000000)
    mf('./Data/clean_position_5M', './Data/clean_test_5M')
    print("done")

