package src.main.java.utils;

import src.main.java.ChessPieces.*;
import java.util.concurrent.*;

public final class TranspositionTable {
    private final ConcurrentMap<Long, TranspositionEntry> table = new ConcurrentHashMap<>();
    
    public void put(long key, int value, int depth, EntryType entryType) {
        table.put(key, new TranspositionEntry(value, depth, entryType));
    }
    
    public TranspositionEntry get(long key) {
        return table.get(key);
    }

    public static class TranspositionEntry {
        public final int value;
        public final int depth;
        public final EntryType entryType;
        
        public TranspositionEntry(int value, int depth, EntryType entryType) {
            this.value = value;
            this.depth = depth;
            this.entryType = entryType;
        }
    }
    
    public enum EntryType {
        EXACT, LOWER_BOUND, UPPER_BOUND
    }
}