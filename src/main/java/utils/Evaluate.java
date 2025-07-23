package src.main.java.utils;

import src.main.java.ChessPieces.Bishop;
import src.main.java.ChessPieces.ChessPiece;
import src.main.java.ChessPieces.King;
import src.main.java.ChessPieces.Knight;
import src.main.java.ChessPieces.Pawn;
import src.main.java.ChessPieces.Queen;
import src.main.java.ChessPieces.Rook;

public class Evaluate {
    private final boolean aiIsWhite;
    private static final int INF = 2000000;
    private static final int MATE_SCORE = 1000000;
    
    // Piece values
    private static final int PAWN_VALUE = 100;
    private static final int KNIGHT_VALUE = 320;
    private static final int BISHOP_VALUE = 330;
    private static final int ROOK_VALUE = 500;
    private static final int QUEEN_VALUE = 900;
    private static final int KING_VALUE = 20000;
    
    // Positional bonuses
    private static final int[][] PAWN_BONUS = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {50, 50, 50, 50, 50, 50, 50, 50},
        {10, 10, 20, 30, 30, 20, 10, 10},
        {5, 5, 10, 25, 25, 10, 5, 5},
        {0, 0, 0, 20, 20, 0, 0, 0},
        {5, -5, -10, 0, 0, -10, -5, 5},
        {5, 10, 10, -20, -20, 10, 10, 5},
        {0, 0, 0, 0, 0, 0, 0, 0}
    };
    public Evaluate(boolean aiIsWhite) {
        this.aiIsWhite = aiIsWhite;
    }
    private int evaluate(ChessPiece[][] board) {
        int total = 0;
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                ChessPiece piece = board[r][c];
                if (piece != null) {
                    int sign = (piece.isWhite == aiIsWhite) ? 1 : -1;
                    total += sign * getPieceValue(piece, r, c);
                }
            }
        }
        return total;
    }

    private int getPieceValue(ChessPiece piece, int row, int col) {
        int value;
        if (piece instanceof Pawn) value = PAWN_VALUE + PAWN_BONUS[row][col];
        else if (piece instanceof Knight) value = KNIGHT_VALUE;
        else if (piece instanceof Bishop) value = BISHOP_VALUE;
        else if (piece instanceof Rook) value = ROOK_VALUE;
        else if (piece instanceof Queen) value = QUEEN_VALUE;
        else if (piece instanceof King) value = KING_VALUE;
        else throw new IllegalArgumentException("Unknown piece");
        
        // Adjust orientation for black pieces
        if (!piece.isWhite) {
            row = 7 - row;
        }
        return value;
    }
}
