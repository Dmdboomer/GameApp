def safe_count_events(file_path):
    count = 0
    with open(file_path, 'r') as file:
        for line in file:
            count += line.count('Event')
            if count % 1000000 == 0:
                print(count)
    return count

# Usage
try:
    event_count = safe_count_events('/Users/yc/Documents/GameEngine/Data/filtered_games.txt')
    print(f"Number of 'Event' occurrences: {event_count}")
except Exception as e:
    print(f"Error occurred: {str(e)}")