package src.main.java.utils;

import src.main.java.ChessPieces.*;
import java.util.Random;

public final class ZobristHasher {
    private static final long[][][] ZOBRIST_TABLE = new long[8][8][12];
    private static final long SIDE_KEY;
    private static final Random RANDOM = new Random();

    static {
        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 8; col++) {
                for (int i = 0; i < 12; i++) {
                    ZOBRIST_TABLE[row][col][i] = RANDOM.nextLong();
                }
            }
        }
        SIDE_KEY = RANDOM.nextLong();
    }

    public static long hashBoard(ChessPiece[][] board, boolean whiteToMove) {
        long hash = whiteToMove ? 0 : SIDE_KEY;
        
        for (int row = 0; row < 8; row++) {
            for (int col = 0; col < 8; col++) {
                ChessPiece piece = board[row][col];
                if (piece != null) {
                    int pieceIndex = pieceToIndex(piece);
                    hash ^= ZOBRIST_TABLE[row][col][pieceIndex];
                }
            }
        }
        return hash;
    }

    private static int pieceToIndex(ChessPiece piece) {
        int base = piece.isWhite ? 0 : 6;
        if (piece instanceof Pawn) return base;
        if (piece instanceof Knight) return base + 1;
        if (piece instanceof Bishop) return base + 2;
        if (piece instanceof Rook) return base + 3;
        if (piece instanceof Queen) return base + 4;
        if (piece instanceof King) return base + 5;
        return -1;
    }
}