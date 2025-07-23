package src.main.java.ChessPieces;
public class Bishop extends ChessPiece {
    public Bishop(boolean isWhite) {
        this.isWhite = isWhite;
        this.symbol = isWhite ? "♗" : "♝";
    }

    @Override
    public boolean isValidMove(Position from, Position to, ChessPiece[][] board) {
        int rowDiff = Math.abs(to.row - from.row);
        int colDiff = Math.abs(to.col - from.col);
        
        // Must move diagonally
        if (rowDiff != colDiff) return false;
        
        // Check for obstacles in path
        int rowStep = to.row > from.row ? 1 : -1;
        int colStep = to.col > from.col ? 1 : -1;
        
        for (int i = 1; i < rowDiff; i++) {
            int row = from.row + i * rowStep;
            int col = from.col + i * colStep;
            if (board[row][col] != null) return false;
        }
        return true;
    }
}