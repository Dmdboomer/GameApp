// src/main/java/ChessGameTwoPlayer.java
package src.main.java;

import javax.swing.*;
import java.awt.*;
import src.main.java.utils.*;
import src.main.java.ChessPieces.*;

public class ChessGameTwoPlayer extends ChessGameBase {

    public ChessGameTwoPlayer() {
        super("Java Chess - 2 Players");
        statusBar.setText("White's turn");
        setVisible(true);
    }

    @Override
    protected void handleSquareClick(Position pos) {
        if (selectedPos == null) {
            handlePieceSelection(pos);
        } else {
            handlePieceMovement(pos);
        }
    }

    private void handlePieceSelection(Position pos) {
        ChessPiece piece = board[pos.row][pos.col];
        if (piece != null && piece.isWhite == isWhiteTurn) {
            selectedPos = pos;
            highlightMoves(pos);
        }
    }

    private void handlePieceMovement(Position pos) {
        ChessPiece selectedPiece = board[selectedPos.row][selectedPos.col];
        
        if (selectedPiece instanceof King && Math.abs(pos.col - selectedPos.col) == 2) {
            handleCastling(pos);
        } else if (helper.isLegalMove(selectedPos, pos, isWhiteTurn)) {
            helper.movePiece(selectedPos, pos);
            endTurn();
        }
        
        checkGameState();
        selectedPos = null;
        renderBoard();
    }

    private void handleCastling(Position pos) {
        boolean isShort = (pos.col - selectedPos.col) > 0;
        if (isShort && helper.canShortCastle(isWhiteTurn)) {
            helper.shortCastle(isWhiteTurn);
            endTurn();
        } else if (!isShort && helper.canLongCastle(isWhiteTurn)) {
            helper.longCastle(isWhiteTurn);
            endTurn();
        }
    }

    private void endTurn() {
        isWhiteTurn = !isWhiteTurn;
        statusBar.setText(isWhiteTurn ? "White's turn" : "Black's turn");
    }

    private void checkGameState() {
        if (helper.detectCheckmate(!isWhiteTurn)) {
            JOptionPane.showMessageDialog(this, "Checkmate!");
        } else if (helper.detectStalemate(!isWhiteTurn)) {
            JOptionPane.showMessageDialog(this, "Stalemate!");
        }
    }
}