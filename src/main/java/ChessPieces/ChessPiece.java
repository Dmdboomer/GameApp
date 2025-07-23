package src.main.java.ChessPieces;
public abstract class ChessPiece {
    public boolean isWhite;
    protected String symbol;
    public boolean hasMoved = false;

    public abstract boolean isValidMove(Position from, Position to, ChessPiece[][] board);

    public String getSymbol() {
        return symbol;
    }
}