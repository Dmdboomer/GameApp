package src.main.java.utils;

import java.util.ArrayList;
import src.main.java.ChessPieces.*;


public class LogicHelper {
    public static final int BOARD_SIZE = 8;
    public ChessPiece[][] board;

    public LogicHelper(ChessPiece[][] board) {
        this.board = board;
    }

    public Position findKingPosition(boolean isWhite, ChessPiece[][] currBoard) {
        for (int row = 0; row < BOARD_SIZE; row++) {
            for (int col = 0; col < BOARD_SIZE; col++) {
                ChessPiece piece = currBoard[row][col];
                if (piece instanceof King && piece.isWhite == isWhite) {
                    return new Position(row, col);
                }
            }
        }
        return null;
    }

    public boolean isInCheck(boolean isWhiteTurn, ChessPiece[][]currBoard){
        Position kingPosition = findKingPosition(isWhiteTurn, currBoard);
        if (kingPosition == null) {
            return false;
        }
        for (int row =0;row<BOARD_SIZE;row++){
            for(int col = 0; col < BOARD_SIZE; col++){
                ChessPiece piece = currBoard[row][col];
                if (piece != null){
                    Position currPos = new Position(row, col);
                    if (piece.isValidMove(currPos, kingPosition, currBoard) &&
                    piece.isWhite == !isWhiteTurn){
                        return true;
                        }
                }
            }
        }
        return false;
    }

    public boolean isValidMove(Position from, Position to) {
        ChessPiece piece = board[from.row][from.col];
        if (piece == null) return false;
        
        ChessPiece targetPiece = board[to.row][to.col];
        if (targetPiece != null && targetPiece.isWhite == piece.isWhite) {
            return false;
        }
        return piece.isValidMove(from, to, board);
    }

    public boolean isLegalMove(Position from, Position to, boolean isWhiteTurn){
        ChessPiece currPiece = board[from.row][from.col];
        if (currPiece == null){
            return false;
        }
        if (currPiece.isWhite != isWhiteTurn){
            return false;
        }
        if (isWhiteTurn){
            if (currPiece instanceof King){
                if (to.row == 7 && to.col == 6 && !currPiece.hasMoved){
                    return canShortCastle(isWhiteTurn);
                } else if (to.row == 7 && to.col == 2 && !currPiece.hasMoved){
                    return canLongCastle(isWhiteTurn);
                }
            }
        } else {
            if (currPiece instanceof King){
                if (to.row == 0 && to.col == 6 && !currPiece.hasMoved){
                    return canShortCastle(isWhiteTurn);
                } else if (to.row == 0 && to.col == 2 && !currPiece.hasMoved){
                    return canLongCastle(isWhiteTurn);
                }
            }
        }
        
        ChessPiece[][] boardCopy = new ChessPiece[BOARD_SIZE][BOARD_SIZE];
        for (int row =0;row<BOARD_SIZE;row++){
            for(int col = 0; col < BOARD_SIZE; col++){
                boardCopy[row][col] = board[row][col];
            }
        }
        boardCopy[to.row][to.col] = board[from.row][from.col];
        boardCopy[from.row][from.col] = null;
        if (isInCheck(isWhiteTurn, boardCopy)){
            return false;
        }
        if (!(isValidMove(from,to))){
            return false;
        }

        return true;
    }

    //FeelsBad ;-;
    public ArrayList<String> generateLegalMoves(boolean isWhiteTurn){
        ArrayList<String> legalMoves = new ArrayList<String>();
    
        // First pass: find all pieces belonging to current player
        ArrayList<Position> pieces = new ArrayList<>();
        for (int row = 0; row < BOARD_SIZE; row++) {
            for (int col = 0; col < BOARD_SIZE; col++) {
                ChessPiece piece = board[row][col];
                if (piece != null && piece.isWhite == isWhiteTurn) {
                    pieces.add(new Position(row, col));
                }
            }
        }
        // Second pass: generate moves for each found piece
        for (Position from : pieces) {
            // Generate possible destinations based on piece type
            for (int row = 0; row < BOARD_SIZE; row++) {
                for (int col = 0; col < BOARD_SIZE; col++) {
                    if (isLegalMove(from, new Position(row,col), isWhiteTurn)) {
                        String move = "From: "+ colToFile(from.row) + rowToChessRow(from.col) + 
                                            " to "+ colToFile(row) + rowToChessRow(col);
                        legalMoves.add(move);
            }
                }
        }
        }
        
        return legalMoves;
    }
    
    public char colToFile(int col){
        return (char)(col + 'a');
    }
    
    public int rowToChessRow(int row){
        return 8-row;
    }

    public void movePiece(Position from, Position to) {
        // Update piece has been moved
        if (board[from.row][from.col]== null);
        else{
            board[from.row][from.col].hasMoved = true;
        }

        // Move it
        board[to.row][to.col] = board[from.row][from.col];
        board[from.row][from.col] = null;

        // Pawn promotion logic
        if (board[to.row][to.col] instanceof Pawn && (to.row == 0 || to.row == 7)) {
            board[to.row][to.col] = new Queen(board[to.row][to.col].isWhite);
        }
        

    }

    public boolean detectCheckmate(boolean isWhiteTurn){
        ArrayList<String> legalMoves = generateLegalMoves(!isWhiteTurn);
        if (legalMoves.size() == 0 && isInCheck(!isWhiteTurn, board)){
            System.out.println("DONE");
            return true;
        }
        return false;
    }
    
    public boolean detectStalemate(boolean isWhiteTurn){
        ArrayList<String> legalMoves = generateLegalMoves(isWhiteTurn);
        if (legalMoves.size() == 0){
            return true;
        }
        return false;
    }
    
    public boolean isSquareAttacked(Position square, boolean attackerIsWhite) {
    for (int r = 0; r < BOARD_SIZE; r++) {
        for (int c = 0; c < BOARD_SIZE; c++) {
            ChessPiece piece = board[r][c];
            if (piece == null || piece.isWhite != attackerIsWhite) continue;
            
            Position from = new Position(r, c);
            if (piece.isValidMove(from, square, board)) {
                return true;
            }
        }
        }
    return false;
    }

    public boolean canShortCastle(boolean isWhiteTurn) {
        int row = isWhiteTurn ? 7 : 0;
        Position kingPos = new Position(row, 4);
        ChessPiece king = board[kingPos.row][kingPos.col];
        if (king == null || !(king instanceof King) || king.hasMoved) return false;
        
        ChessPiece rook = board[row][7];
        if (rook == null || !(rook instanceof Rook) || rook.hasMoved) return false;
        
        // Check squares between are empty
        if (board[row][5] != null || board[row][6] != null) return false;
        
        // King not in check
        if (isInCheck(isWhiteTurn, board)) return false;
        
        // Check intermediate squares
        if (isSquareAttacked(new Position(row, 5), !isWhiteTurn) ||
            isSquareAttacked(new Position(row, 6), !isWhiteTurn)) {
            return false;
        }
        
        return true;
    }

    public boolean canLongCastle(boolean isWhiteTurn) {
        int row = isWhiteTurn ? 7 : 0;
        Position kingPos = new Position(row, 4);
        ChessPiece king = board[kingPos.row][kingPos.col];
        if (king == null || !(king instanceof King) || king.hasMoved) return false;
        
        ChessPiece rook = board[row][0];
        if (rook == null || !(rook instanceof Rook) || rook.hasMoved) return false;
        
        // Check squares between are empty
        if (board[row][1] != null || board[row][2] != null || board[row][3] != null) return false;
        
        // King not in check
        if (isInCheck(isWhiteTurn, board)) return false;
        
        // Check intermediate squares
        if (isSquareAttacked(new Position(row, 3), !isWhiteTurn) ||
            isSquareAttacked(new Position(row, 2), !isWhiteTurn)) {
            return false;
        }
        
        return true;
    }
    
    public void shortCastle(boolean isWhiteTurn){
        if (isWhiteTurn){
            Position e1 = new Position(7, 4);
            Position f1 = new Position(7, 5);
            Position g1 = new Position(7, 6);
            Position h1 = new Position(7, 7);
            movePiece(e1,g1);
            movePiece(h1,f1);
        } else {
            Position e8 = new Position(0, 4);
            Position f8 = new Position(0, 5);
            Position g8 = new Position(0, 6);
            Position h8 = new Position(0, 7);
            movePiece(e8,g8);
            movePiece(h8,f8);
        }
    }
    
    public void longCastle(boolean isWhiteTurn){
        if (isWhiteTurn){
            Position a1 = new Position(7, 0);
            Position c1 = new Position(7, 2);
            Position d1 = new Position(7, 3);
            Position e1 = new Position(7, 4);
            movePiece(e1,c1);
            movePiece(a1,d1);
        } else {
            Position a8 = new Position(0, 0);
            Position c8 = new Position(0, 2);
            Position d8 = new Position(0, 3);
            Position e8 = new Position(0, 4);
            movePiece(e8,c8);
            movePiece(a8,d8);
        }
    }

    public static String boardToFEN(ChessPiece[][]board, boolean isWhiteTurn){
        String result = "";
        for (int row =0; row < 8; row++){
            int currSpace = 0;
            String fenRow = "";
            for (int col = 0; col < 8; col++){
                boolean isWhitePiece;
                ChessPiece currPiece = board[row][col];
                if (currPiece == null){
                    currSpace+=1;
                } else {
                    if (currPiece.isWhite){
                        isWhitePiece = true;
                    } else {
                        isWhitePiece = false;
                    }
                    if (currSpace > 0){
                        fenRow += String.valueOf(currSpace);
                        currSpace = 0;
                    }
                    fenRow += pieceToSymbol(currPiece, isWhitePiece);
                }
            }

            if (currSpace > 0){
                fenRow += String.valueOf(currSpace);
                currSpace = 0;
            }
            result += fenRow;
            if (row <= 6){
                result += "/";
            }
            fenRow = "";
        }
        result += " ";

        if (isWhiteTurn){
            result += "w";
        } else{
            result += "b";
        }
        result += " ";

        ChessPiece a1 = board[7][0];
        ChessPiece a8 = board[0][0];
        ChessPiece e1 = board[7][4];
        ChessPiece e8 = board[0][4];
        ChessPiece h1 = board[7][7];
        ChessPiece h8 = board[0][7];
        boolean couldShortCastleW = true;
        boolean couldShortCastleB = true;
        boolean couldLongCastleW = true;
        boolean couldLongCastleB = true;
        if (!(a1 instanceof Rook) || !(e1 instanceof King) || a1.hasMoved || e1.hasMoved){
            couldLongCastleW = false;
        }
        if (!(a8 instanceof Rook) || !(e8 instanceof King) || a8.hasMoved || e8.hasMoved){
            couldLongCastleB = false;
        }
        if (!(h1 instanceof Rook) || !(e1 instanceof King) || h1.hasMoved || e1.hasMoved){
            couldShortCastleW = false;
        }
        if (!(h8 instanceof Rook) || !(e8 instanceof King) || a1.hasMoved || e1.hasMoved){
            couldShortCastleB = false;
        }
        if (!couldLongCastleB && !couldLongCastleW && !couldShortCastleB && ! couldShortCastleW){
            result += "-";
        } else{
            if (couldShortCastleW){
                result += "K";
            }
            if (couldLongCastleW){
                result += "Q";
            }
            if (couldShortCastleB){
                result += "k";
            }
            if (couldLongCastleB){
                result += "q";
            }
        }
        
        // Need to do en pessant stuff
        result += " - 0 0";
        

        return result;
    }
    
    public static String pieceToSymbol(ChessPiece piece, boolean isWhite){
        String symbol;
        if (piece instanceof Pawn){
            symbol = "p";
        } else if (piece instanceof King){
            symbol = "k";
        } else if (piece instanceof Knight){
            symbol = "n";
        } else if (piece instanceof Queen){
            symbol = "q";
        } else if (piece instanceof Bishop){
            symbol = "b";
        } else{
            symbol = "r";
        }
        if (isWhite){
            symbol = symbol.toUpperCase();
        }
        return symbol;

    }

    public static void main(String[] args) {
        ChessPiece[][] board = new ChessPiece[BOARD_SIZE][BOARD_SIZE];

        for (int col = 0; col < BOARD_SIZE; col++) {
            board[1][col] = new Pawn(false);
            board[6][col] = new Pawn(true);
        }
        
        // Black pieces
        board[0][0] = new Rook(false);
        board[0][7] = new Rook(false);
        board[0][1] = new Knight(false);
        board[0][6] = new Knight(false);
        board[0][2] = new Bishop(false);
        board[0][5] = new Bishop(false);
        board[0][3] = new Queen(false);
        board[0][4] = new King(false);
        
        // White pieces
        board[7][0] = new Rook(true);
        board[7][7] = new Rook(true);
        board[7][1] = new Knight(true);
        board[7][6] = new Knight(true);
        board[7][2] = new Bishop(true);
        board[7][5] = new Bishop(true);
        board[7][3] = new Queen(true);
        board[7][4] = new King(true);

        board[6][3] = null;
        board[1][2] = null;
        board[3][2] = new Pawn(true);

        System.out.println(boardToFEN(board, true));
    }
}
