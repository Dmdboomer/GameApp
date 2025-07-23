package src.main.java.ChessPieces;
public class Queen extends ChessPiece {
    public Queen(boolean isWhite) {
        this.isWhite = isWhite;
        this.symbol = isWhite ? "♕" : "♛";
    }

    @Override
    public boolean isValidMove(Position from, Position to, ChessPiece[][] board) {
        int rowDiff = Math.abs(to.row - from.row);
        int colDiff = Math.abs(to.col - from.col);
        
        // Like a rook
        if (from.row == to.row || from.col == to.col) {
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
        // Like a bishop
        else if (rowDiff == colDiff) {
            int rowStep = to.row > from.row ? 1 : -1;
            int colStep = to.col > from.col ? 1 : -1;
            for (int i = 1; i < rowDiff; i++) {
                int row = from.row + i * rowStep;
                int col = from.col + i * colStep;
                if (board[row][col] != null) return false;
            }
            return true;
        }
        return false;
    }
}