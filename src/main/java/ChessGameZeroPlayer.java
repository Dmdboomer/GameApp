package src.main.java;

import javax.swing.*;
import java.awt.*;
import src.main.java.utils.*;
import src.main.java.ChessPieces.*;
import src.main.java.ChessGameOnePlayer;

import static src.main.java.utils.LogicHelper.boardToFEN;


public class ChessGameZeroPlayer extends ChessGameBase {
    private final AlphaBetaPruning whiteAI;
    private final NNAI blackAI;
    private Timer aiMoveTimer;
    
    public ChessGameZeroPlayer() {
        super("Chess - AI vs AI");
        whiteAI = new AlphaBetaPruning(true);
        blackAI = new NNAI();
        
        initializeAITimer();
        startGame();
    }

    // python bot
    private static class NNAI {
        public String findBestMove(ChessPiece[][] board, boolean isWhiteTurn) {
            String fen = boardToFEN(board, isWhiteTurn);
            try {
                String[] moves = ChessAIPythonCaller.getBestMove(fen);
                return moves[0];
            } catch (Exception e) {
                e.printStackTrace();
                return null;
            }
        }
    }
    
    private void initializeAITimer() {
        aiMoveTimer = new Timer(1000, e -> makeAIMove());
        aiMoveTimer.setRepeats(true);
    }
    
    private void startGame() {
        statusBar.setText("AI vs AI - Game started");
        aiMoveTimer.start();
    }
    
    // Remove human interaction
    @Override
    protected void handleSquareClick(Position pos) {}
    
    private void makeAIMove() {
        aiMoveTimer.stop();
        statusBar.setText("AI thinking... (" + (isWhiteTurn ? "White" : "Black") + ")");
        
        new SwingWorker<Void, Void>() {
            String aiMove;
            
            @Override
            protected Void doInBackground() {
                // Choose AI based on current turn
                if (isWhiteTurn) {
                    aiMove = whiteAI.findBestMove(board, isWhiteTurn);
                } else {
                    aiMove = blackAI.findBestMove(board, isWhiteTurn);
                }
                return null;
            }
            
            @Override
            protected void done() {
                if (aiMove == null || aiMove.length() < 4) {
                    gameOver("Game over - " + (isWhiteTurn ? "White" : "Black") + " cannot move");
                    return;
                }
                
                // Parse move positions
                Position from = new Position(
                    8 - Character.getNumericValue(aiMove.charAt(1)),
                    aiMove.charAt(0) - 'a'
                );
                Position to = new Position(
                    8 - Character.getNumericValue(aiMove.charAt(3)),
                    aiMove.charAt(2) - 'a'
                );
                
                doMove(from, to);
                renderBoard();
                
                if (helper.detectCheckmate(isWhiteTurn)) {
                    gameOver((isWhiteTurn ? "Black" : "White") + " wins by checkmate!");
                } else if (helper.detectStalemate(isWhiteTurn)) {
                    gameOver("Stalemate! Game drawn.");
                } else {
                    isWhiteTurn = !isWhiteTurn;
                    aiMoveTimer.start();
                    statusBar.setText("Turn: " + (isWhiteTurn ? "White" : "Black"));
                }
            }
        }.execute();
    }
    
    private void gameOver(String message) {
        JOptionPane.showMessageDialog(this, message);
        statusBar.setText("Game Over");
    }

    // Reusing existing doMove implementation
    protected void doMove(Position from, Position to) {
        System.out.println(boardToFEN(board, isWhiteTurn));
        ChessPiece piece = board[from.row][from.col];
        board[to.row][to.col] = piece;
        board[from.row][from.col] = null;
        piece.hasMoved = true;

        // Castling logic
        if (piece instanceof King) {
            int colDiff = to.col - from.col;
            if (Math.abs(colDiff) == 2) {
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
        
        // Pawn promotion
        if (piece instanceof Pawn && (to.row == 0 || to.row == 7)) {
            board[to.row][to.col] = new Queen(piece.isWhite);
        }

        helper = new LogicHelper(board);
    }
}