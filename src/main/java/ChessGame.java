package src.main.java;
import javax.swing.*;

import src.main.java.ChessAIPythonCaller;
import src.main.java.ChessPieces.Bishop;
import src.main.java.ChessPieces.ChessPiece;
import src.main.java.ChessPieces.King;
import src.main.java.ChessPieces.Knight;
import src.main.java.ChessPieces.Pawn;
import src.main.java.ChessPieces.Position;
import src.main.java.ChessPieces.Queen;
import src.main.java.ChessPieces.Rook;

import src.main.java.utils.AlphaBetaPruning;
import src.main.java.utils.LogicHelper;

import java.awt.*;
import java.awt.event.*;
import java.util.ArrayList;


public class ChessGame extends JFrame {
    private static final int BOARD_SIZE = 8;
    private final JPanel boardPanel = new JPanel(new GridLayout(BOARD_SIZE, BOARD_SIZE));
    private final JLabel statusBar = new JLabel("White's turn");
    private ChessPiece[][] board = new ChessPiece[BOARD_SIZE][BOARD_SIZE];
    private Position selectedPos = null;
    private boolean isWhiteTurn = true;

    public ChessGame() {
        super("Java Chess");
        initializeBoard();
        setupUI();
    }

    private void initializeBoard() {
        // Initialize pawns
        for (int col = 0; col < BOARD_SIZE; col++) {
            board[1][col] = new Pawn(false);
            board[6][col] = new Pawn(true);
        }
        
        // Back row pieces for black
        board[0][0] = new Rook(false);
        board[0][7] = new Rook(false);
        board[0][1] = new Knight(false);
        board[0][6] = new Knight(false);
        board[0][2] = new Bishop(false);
        board[0][5] = new Bishop(false);
        board[0][3] = new Queen(false);
        board[0][4] = new King(false);
        
        // Back row pieces for white
        board[7][0] = new Rook(true);
        board[7][7] = new Rook(true);
        board[7][1] = new Knight(true);
        board[7][6] = new Knight(true);
        board[7][2] = new Bishop(true);
        board[7][5] = new Bishop(true);
        board[7][3] = new Queen(true);
        board[7][4] = new King(true);
    }

    private void setupUI() {
        setSize(600, 600);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());
        
        boardPanel.setPreferredSize(new Dimension(500, 500));
        add(boardPanel, BorderLayout.CENTER);
        add(statusBar, BorderLayout.SOUTH);
        
        renderBoard();
        setVisible(true);
    }

    private void renderBoard() {
        boardPanel.removeAll();
        for (int row = 0; row < BOARD_SIZE; row++) {
            for (int col = 0; col < BOARD_SIZE; col++) {
                JButton square = new JButton();
                square.setOpaque(true);
                square.setBorderPainted(false);
                square.setBackground((row + col) % 2 == 0 ? Color.WHITE : Color.GRAY);
                if (board[row][col] != null) {
                    square.setText(board[row][col].getSymbol());
                    square.setFont(new Font("Arial", Font.BOLD, 24));
                }
                
                final Position pos = new Position(row, col);
                square.addActionListener(e -> handleSquareClick(pos));
                boardPanel.add(square);
            }
        }
        boardPanel.revalidate();
        boardPanel.repaint();
    }

    private void handleSquareClick(Position pos) {
        LogicHelper helper = new LogicHelper(board);

        
        // Ai stuff
        AlphaBetaPruning ai = new AlphaBetaPruning(isWhiteTurn);
        String bestMove = ai.findBestMove(board, isWhiteTurn);
        String result = bestMove;
        System.out.println(result);
        
        
        if (selectedPos == null) {
            ChessPiece piece = board[pos.row][pos.col];
            if (piece != null && piece.isWhite == isWhiteTurn) {
                selectedPos = pos;
                highlightMoves(pos);
            }
        } else {
            ChessPiece selectedPiece = board[selectedPos.row][selectedPos.col];
            // Check for castling
            if (selectedPiece instanceof King &&
                Math.abs(pos.col - selectedPos.col) == 2 &&
                pos.row == selectedPos.row) {
                
                boolean isShort = (pos.col - selectedPos.col) > 0;
                if (isShort && helper.canShortCastle(isWhiteTurn)) {
                    helper.shortCastle(isWhiteTurn);
                    isWhiteTurn = !isWhiteTurn;
                    statusBar.setText(isWhiteTurn ? "White's turn" : "Black's turn");
                } 
                else if (!isShort && helper.canLongCastle(isWhiteTurn)) {
                    helper.longCastle(isWhiteTurn);
                    isWhiteTurn = !isWhiteTurn;
                    statusBar.setText(isWhiteTurn ? "White's turn" : "Black's turn");
                }
            }
            // Normal moves
            else if (helper.isLegalMove(selectedPos, pos, isWhiteTurn)) {
                helper.movePiece(selectedPos, pos);
                isWhiteTurn = !isWhiteTurn;
                statusBar.setText(isWhiteTurn ? "White's turn" : "Black's turn");
            }
            
            // Check game state
            if (helper.detectCheckmate(!isWhiteTurn)) {
                JOptionPane.showMessageDialog(this, "Checkmate!");
            } else if (helper.detectStalemate(!isWhiteTurn)) {
                JOptionPane.showMessageDialog(this, "Stalemate!");
            }
            
            selectedPos = null;
            renderBoard();

        }
    }

    private void highlightMoves(Position from) {
        // Implementation for highlighting possible moves would go here
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(ChessGame::new);
    }

}