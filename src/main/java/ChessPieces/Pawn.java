package src.main.java.ChessPieces;
public class Pawn extends ChessPiece {
    public Pawn(boolean isWhite) {
        this.isWhite = isWhite;
        this.symbol = isWhite ? "♙" : "♟";
    }

    @Override
    public boolean isValidMove(Position from, Position to, ChessPiece[][] board) {
        int direction = isWhite ? -1 : 1;
        int startRow = isWhite ? 6 : 1;
        
        // Forward move
        if (from.col == to.col) {
            // Single step
            if (to.row == from.row + direction) {
                return board[to.row][to.col] == null;
            }
            // Double step from start position
            if (from.row == startRow && to.row == from.row + 2 * direction) {
                return board[from.row + direction][from.col] == null && 
                       board[to.row][to.col] == null;
            }
        }
        // Capture move
        else if (Math.abs(from.col - to.col) == 1 && to.row == from.row + direction) {
            return board[to.row][to.col] != null || 
                   enPassantPossible(from, to, board);
        }
        return false;
    }
    
    private boolean enPassantPossible(Position from, Position to, ChessPiece[][] board) {
        // En passant logic would be implemented here
        return false;
    }
}