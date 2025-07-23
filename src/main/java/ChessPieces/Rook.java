package src.main.java.ChessPieces;
public class Rook extends ChessPiece {
    public Rook(boolean isWhite) {
        this.isWhite = isWhite;
        this.symbol = isWhite ? "♖" : "♜";
    }

    @Override
    public boolean isValidMove(Position from, Position to, ChessPiece[][] board) {
        // Must move horizontally or vertically
        if (from.row != to.row && from.col != to.col) return false;
        
        // Check for obstacles in path
        if (from.row == to.row) {
            int step = to.col > from.col ? 1 : -1;
            for (int col = from.col + step; col != to.col; col += step) {
                if (board[from.row][col] != null) return false;
            }
        } else {
            int step = to.row > from.row ? 1 : -1;
            for (int row = from.row + step; row != to.row; row += step) {
                if (board[row][from.col] != null) return false;
            }
        }
        return true;
    }
}