package src.main.java;
import javax.swing.*;

import java.awt.*;
import java.awt.event.*;

public class ChessMenu extends JFrame {
    public ChessMenu() {
        super("Chess Menu");
        setupUI();
    }

    private void setupUI() {
        setSize(400, 300);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new GridLayout(3, 1));

        JButton twoPlayerButton = createMenuButton("2 Player", Color.GREEN);
        JButton vsAIButton = createMenuButton("1 Player", Color.BLUE);
        JButton AIvsAIButton = createMenuButton("0 Player???", Color.YELLOW);
        JButton exitButton = createMenuButton("Exit", Color.RED);

        twoPlayerButton.addActionListener(e -> startChessGame2p());
        vsAIButton.addActionListener(e -> startChessGame1p());
        AIvsAIButton.addActionListener(e -> startChessGame0p());
        exitButton.addActionListener(e -> System.exit(0));
        
        add(twoPlayerButton);
        add(vsAIButton);
        add(AIvsAIButton);
        add(exitButton);

        setLocationRelativeTo(null);
        setVisible(true);
    }

    private JButton createMenuButton(String text, Color bgColor) {
        JButton button = new JButton(text);
        button.setFont(new Font("Arial", Font.BOLD, 24));
        button.setBackground(bgColor);
        button.setForeground(Color.WHITE);
        button.setFocusPainted(false);
        button.setBorder(BorderFactory.createLineBorder(Color.BLACK, 2));
        button.setPreferredSize(new Dimension(150, 60));
        return button;
    }

    private void startChessGame2p() {
        dispose();
        SwingUtilities.invokeLater(() -> new ChessGameTwoPlayer().setVisible(true));
    }
    private void startChessGame1p() {
        dispose();
        SwingUtilities.invokeLater(() -> new ChessGameOnePlayer(true, 0).setVisible(true));
    }
    private void startChessGame0p() {
        dispose();
        SwingUtilities.invokeLater(() -> new ChessGameZeroPlayer().setVisible(true));
    }

    private void showComingSoon() {
        JOptionPane.showMessageDialog(this,
            "AI feature coming soon!",
            "Under Development",
            JOptionPane.INFORMATION_MESSAGE);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(ChessMenu::new);
    }
}