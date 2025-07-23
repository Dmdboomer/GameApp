package src.main.java.ChessPieces;
public class Knight extends ChessPiece {
    public Knight(boolean isWhite) {
        this.isWhite = isWhite;
        this.symbol = isWhite ? "♘" : "♞";
    }

    @Override
    public boolean isValidMove(Position from, Position to, ChessPiece[][] board) {
        int rowDiff = Math.abs(to.row - from.row);
        int colDiff = Math.abs(to.col - from.col);
        return (rowDiff == 2 && colDiff == 1) || (rowDiff == 1 && colDiff == 2);
    }
}