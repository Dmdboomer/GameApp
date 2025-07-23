package src.main.java;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import src.main.java.utils.*;
import src.main.java.ChessPieces.*;

public class ChessGameOnePlayer extends ChessGameBase {
    private final boolean humanIsWhite;
    private final AlphaBetaPruning ai;
    private boolean isAITurn = false;
    private JButton aiMoveButton;
    private int botSelection;

    public ChessGameOnePlayer(boolean humanIsWhite, int botSelection) {
        super("Chess - Human vs AI");
        this.humanIsWhite = humanIsWhite;
        this.ai = new AlphaBetaPruning(!humanIsWhite);
        this.botSelection = botSelection;
        isWhiteTurn = humanIsWhite;
        isAITurn = !humanIsWhite;
        
        // Create "AI Move" button
        aiMoveButton = new JButton("AI Move");
        aiMoveButton.setEnabled(isAITurn);
        aiMoveButton.addActionListener(e -> makeAIMove());
        
        // Add button to control panel
        JPanel buttonPanel = new JPanel();
        buttonPanel.add(aiMoveButton);
        add(buttonPanel, BorderLayout.SOUTH);
    }

    @Override
    protected void handleSquareClick(Position pos) {
        if (isAITurn) return;

        if (selectedPos == null) {
            ChessPiece piece = board[pos.row][pos.col];
            if (piece != null && piece.isWhite == isWhiteTurn) {
                selectedPos = pos;
                highlightMoves(pos);
            }
        } else {
            Position from = selectedPos;
            Position to = pos;
            
            if (helper.isLegalMove(from, to, isWhiteTurn)) {
                doMove(from, to);
                selectedPos = null;
                renderBoard();
                
                if (helper.detectCheckmate(isWhiteTurn)) {
                    JOptionPane.showMessageDialog(this, 
                        (isWhiteTurn ? "White" : "Black") + " wins by checkmate!");
                } else if (helper.detectStalemate(isWhiteTurn)) {
                    JOptionPane.showMessageDialog(this, "Stalemate! Game drawn.");
                } else {
                    // Switch turns and enable AI button
                    isWhiteTurn = !isWhiteTurn;
                    isAITurn = !isAITurn;
                    aiMoveButton.setEnabled(isAITurn);
                    statusBar.setText("Turn: " + (isWhiteTurn ? "White" : "Black") +
                                     (isAITurn ? " - Click 'AI Move'" : ""));
                }
            } else {
                // Select new piece
                ChessPiece piece = board[pos.row][pos.col];
                if (piece != null && piece.isWhite == isWhiteTurn) {
                    selectedPos = pos;
                    highlightMoves(pos);
                } else {
                    selectedPos = null;
                    renderBoard();
                }
            }
        }
    }

    private void makeAIMove() {
        statusBar.setText("AI is thinking...");
        aiMoveButton.setEnabled(false);
        
        new SwingWorker<Void, Void>() {
            String aiMove;
            
            @Override
            protected Void doInBackground() {
                if (botSelection == 0){
                    aiMove = ai.findBestMove(board, isWhiteTurn);
                }
                else if ( 0 < botSelection  && botSelection <2){
                    aiMove = "e3e2";
                }
                return null;
            }
            
            @Override
            protected void done() {
                if (aiMove != null && aiMove.length() == 4) {
                    Position from = new Position(
                        8 - Character.getNumericValue(aiMove.charAt(1)),
                        aiMove.charAt(0) - 'a'
                    );
                    Position to = new Position(
                        8 - Character.getNumericValue(aiMove.charAt(3)),
                        aiMove.charAt(2) - 'a'
                    );
                    
                    doMove(from, to);
                    
                    if (helper.detectCheckmate(isWhiteTurn)) {
                        JOptionPane.showMessageDialog(ChessGameOnePlayer.this, 
                            "AI wins by checkmate!");
                    } else if (helper.detectStalemate(isWhiteTurn)) {
                        JOptionPane.showMessageDialog(ChessGameOnePlayer.this, "Stalemate! Game drawn.");
                    } else {
                        // Switch turns and update UI
                        isWhiteTurn = !isWhiteTurn;
                        isAITurn = !isAITurn;
                        aiMoveButton.setEnabled(isAITurn);
                    }
                } else if (aiMove != null) {
                    JOptionPane.showMessageDialog(ChessGameOnePlayer.this, 
                        "Game over - " + (isWhiteTurn ? "White" : "Black") + " cannot move");
                }
                
                statusBar.setText("Turn: " + (isWhiteTurn ? "White" : "Black") +
                                 (isAITurn ? " - Click 'AI Move'" : ""));
                selectedPos = null;
                renderBoard();
            }
        }.execute();
    }
    
    // Enhanced move execution with castling and promotion support
    public void doMove(Position from, Position to) {
        ChessPiece piece = board[from.row][from.col];
        board[to.row][to.col] = piece;
        board[from.row][from.col] = null;
        piece.hasMoved = true;

        // Handle castling
        if (piece instanceof King) {
            int colDiff = to.col - from.col;
            if (Math.abs(colDiff) == 2) { // Castling move
                int rookFromCol, rookToCol;
                if (colDiff > 0) { // Kingside
                    rookFromCol = 7;
                    rookToCol = 5;
                } else { // Queenside
                    rookFromCol = 0;
                    rookToCol = 3;
                }
                
                ChessPiece rook = board[to.row][rookFromCol];
                board[to.row][rookToCol] = rook;
                board[to.row][rookFromCol] = null;
                if (rook != null) rook.hasMoved = true;
            }
        }
        
        // Handle pawn promotion
        if (piece instanceof Pawn && (to.row == 0 || to.row == 7)) {
            board[to.row][to.col] = new Queen(piece.isWhite);
        }

        
        // Update helper with new board state
        helper = new LogicHelper(board);
    }
}