// src/main/java/ChessGameBase.java
package src.main.java;

import javax.swing.*;
import java.awt.*;
import src.main.java.utils.*;
import src.main.java.ChessPieces.*;

public abstract class ChessGameBase extends JFrame {
    protected static final int BOARD_SIZE = 8;
    protected JPanel boardPanel = new JPanel(new GridLayout(BOARD_SIZE, BOARD_SIZE));
    protected JLabel statusBar = new JLabel();
    protected ChessPiece[][] board = new ChessPiece[BOARD_SIZE][BOARD_SIZE];
    protected Position selectedPos = null;
    protected boolean isWhiteTurn = true;
    protected LogicHelper helper;

    public ChessGameBase(String title) {
        super(title);
        initializeBoard();
        helper = new LogicHelper(board);
        setupBoardUI();
    }

    protected void initializeBoard() {
        // Pawns
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
    }

    protected void setupBoardUI() {
        setSize(600, 600);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());
        
        boardPanel.setPreferredSize(new Dimension(500, 500));
        add(boardPanel, BorderLayout.CENTER);
        add(statusBar, BorderLayout.SOUTH);
        
        renderBoard();
    }

    protected void renderBoard() {
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

    protected abstract void handleSquareClick(Position pos);

    protected void highlightMoves(Position from) {
        // Implementation for highlighting moves
    }
}