package src.main.java.utils;

import src.main.java.utils.TranspositionTable;
import src.main.java.utils.ZobristHasher;
import src.main.java.utils.LogicHelper;
import src.main.java.ChessPieces.*;
import java.util.*;

public final class AlphaBetaPruning {
    private final boolean aiIsWhite;
    private final TranspositionTable transTable = new TranspositionTable();
    private static final int MAX_DEPTH = 1;
    private static final int INF = 2000000;
    private static final int MATE_SCORE = 1000000;
    private static final int WIN_SCORE = 900000;
    
    // Piece values
    private static final int PAWN_VALUE = 100;
    private static final int KNIGHT_VALUE = 320;
    private static final int BISHOP_VALUE = 330;
    private static final int ROOK_VALUE = 500;
    private static final int QUEEN_VALUE = 900;
    private static final int KING_VALUE = 20000;
    
    // Positional bonuses
    private static final int[][] PAWN_BONUS = {
        {0, 0, 0, 0, 0, 0, 0, 0},
        {50, 50, 50, 50, 50, 50, 50, 50},
        {10, 10, 20, 30, 30, 20, 10, 10},
        {5, 5, 10, 25, 25, 10, 5, 5},
        {0, 0, 0, 20, 20, 0, 0, 0},
        {5, -5, -10, 0, 0, -10, -5, 5},
        {5, 10, 10, -20, -20, 10, 10, 5},
        {0, 0, 0, 0, 0, 0, 0, 0}
    };
    
    public AlphaBetaPruning(boolean aiIsWhite) {
        this.aiIsWhite = aiIsWhite;
    }
    
    public String findBestMove(ChessPiece[][] board, boolean currentTurn) {
        if (currentTurn != aiIsWhite) {
            throw new IllegalArgumentException("AI should only be called on its turn");
        }
        
        LogicHelper helper = new LogicHelper(copyBoard(board));
        List<Move> moves = generateAllLegalMoves(helper, aiIsWhite);
        if (moves.isEmpty()) return null;
        
        // Track best moves across iterations (for deepest depth)
        List<Move> overallBestMoves = new ArrayList<>();
        int overallBestValue = -INF;

        // Iterative deepening
        for (int depth = 1; depth <= MAX_DEPTH; depth++) {
            int alpha = -INF;
            int beta = INF;
            int currentBestValue = -INF;
            List<Move> currentDepthBestMoves = new ArrayList<>();
            
            for (Move move : moves) {
                ChessPiece[][] newBoard = makeMove(board, move);
                int value = alphaBeta(newBoard, !aiIsWhite, depth-1, alpha, beta);
                
                // Track best moves for current depth
                if (value > currentBestValue) {
                    currentBestValue = value;
                    currentDepthBestMoves.clear();
                    currentDepthBestMoves.add(move);
                } else if (value == currentBestValue) {
                    currentDepthBestMoves.add(move);
                }
                
                // Alpha-beta updates
                if (value > alpha) alpha = value;
                if (alpha >= beta) break; // Beta cutoff
            }
            
            // Update overall best moves/value for deepest depth
            if (!currentDepthBestMoves.isEmpty()) {
                overallBestValue = currentBestValue;
                overallBestMoves = currentDepthBestMoves;
            }
        }
        
        // Random selection among best moves
        if (overallBestMoves.isEmpty()) return null;
        Random rand = new Random();
        Move bestMove = overallBestMoves.get(rand.nextInt(overallBestMoves.size()));
        return moveToString(bestMove);
    }
    
    private int alphaBeta(ChessPiece[][] board, boolean currentTurn, int depth, int alpha, int beta) {
        long hash = ZobristHasher.hashBoard(board, currentTurn);
        TranspositionTable.TranspositionEntry entry = transTable.get(hash);
        
        // Transposition table lookup
        if (entry != null && entry.depth >= depth) {
            switch (entry.entryType) {
                case EXACT: return entry.value;
                case LOWER_BOUND: alpha = Math.max(alpha, entry.value); break;
                case UPPER_BOUND: beta = Math.min(beta, entry.value); break;
            }
            if (alpha >= beta) return entry.value;
        }
        
        LogicHelper helper = new LogicHelper(board);
        
        // Check terminal states
        if (helper.detectCheckmate(currentTurn)) {
            return (currentTurn == aiIsWhite) ? -(MATE_SCORE - depth) : (MATE_SCORE - depth);
        }
        if (helper.detectStalemate(currentTurn)) {
            return 0;
        }
        
        // Leaf node condition
        if (depth <= 0) {
            return quiesce(board, currentTurn, alpha, beta);
        }
        
        List<Move> moves = generateAllLegalMoves(helper, currentTurn);
        TranspositionTable.EntryType entryType = TranspositionTable.EntryType.EXACT;
        int value;
        
        if (currentTurn == aiIsWhite) {  // Maximizing player
            value = -INF;
            for (Move move : moves) {
                ChessPiece[][] newBoard = makeMove(board, move);
                int score = alphaBeta(newBoard, !currentTurn, depth-1, alpha, beta);
                value = Math.max(value, score);
                alpha = Math.max(alpha, value);
                if (alpha >= beta) {
                    entryType = TranspositionTable.EntryType.LOWER_BOUND;
                    break;
                }
            }
        } else {  // Minimizing player
            value = INF;
            for (Move move : moves) {
                ChessPiece[][] newBoard = makeMove(board, move);
                int score = alphaBeta(newBoard, !currentTurn, depth-1, alpha, beta);
                value = Math.min(value, score);
                beta = Math.min(beta, value);
                if (alpha >= beta) {
                    entryType = TranspositionTable.EntryType.UPPER_BOUND;
                    break;
                }
            }
        }
        
        transTable.put(hash, value, depth, entryType);
        return value;
    }
    
    private int quiesce(ChessPiece[][] board, boolean currentTurn, int alpha, int beta) {
        int standPat = evaluate(board);
        if (currentTurn == aiIsWhite) {
            if (standPat >= beta) return beta;
            alpha = Math.max(alpha, standPat);
        } else {
            if (standPat <= alpha) return alpha;
            beta = Math.min(beta, standPat);
        }
        
        LogicHelper helper = new LogicHelper(board);
        List<Move> captures = generateCaptures(helper, currentTurn);
        
        for (Move move : captures) {
            ChessPiece[][] newBoard = makeMove(board, move);
            int score = quiesce(newBoard, !currentTurn, alpha, beta);
            
            if (currentTurn == aiIsWhite) {
                if (score >= beta) return beta;
                alpha = Math.max(alpha, score);
            } else {
                if (score <= alpha) return alpha;
                beta = Math.min(beta, score);
            }
        }
        
        return (currentTurn == aiIsWhite) ? alpha : beta;
    }
    
    private ChessPiece[][] copyBoard(ChessPiece[][] board) {
        ChessPiece[][] copy = new ChessPiece[8][8];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                copy[i][j] = copyPiece(board[i][j]);
            }
        }
        return copy;
    }
    
    private ChessPiece copyPiece(ChessPiece piece) {
        if (piece == null) return null;
        
        ChessPiece copy;
        if (piece instanceof Pawn) {
            copy = new Pawn(piece.isWhite);
        } else if (piece instanceof Knight) {
            copy = new Knight(piece.isWhite);
        } else if (piece instanceof Bishop) {
            copy = new Bishop(piece.isWhite);
        } else if (piece instanceof Rook) {
            copy = new Rook(piece.isWhite);
        } else if (piece instanceof Queen) {
            copy = new Queen(piece.isWhite);
        } else if (piece instanceof King) {
            copy = new King(piece.isWhite);
        } else {
            throw new IllegalArgumentException("Unknown piece type");
        }
        copy.hasMoved = piece.hasMoved;
        return copy;
    }
    
    private List<Move> generateAllLegalMoves(LogicHelper helper, boolean isWhiteTurn) {
        List<Move> moves = new ArrayList<>();
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                ChessPiece piece = helper.board[r][c];
                if (piece != null && piece.isWhite == isWhiteTurn) {
                    for (int toR = 0; toR < 8; toR++) {
                        for (int toC = 0; toC < 8; toC++) {
                            Position from = new Position(r, c);
                            Position to = new Position(toR, toC);
                            if (helper.isLegalMove(from, to, isWhiteTurn)) {
                                moves.add(new Move(from, to));
                            }
                        }
                    }
                }
            }
        }
        
        // Castling moves
        int row = isWhiteTurn ? 7 : 0;
        if (helper.canShortCastle(isWhiteTurn)) {
            moves.add(new Move(new Position(row, 4), new Position(row, 6)));
        }
        if (helper.canLongCastle(isWhiteTurn)) {
            moves.add(new Move(new Position(row, 4), new Position(row, 2)));
        }
        
        return moves;
    }
    
    private List<Move> generateCaptures(LogicHelper helper, boolean isWhiteTurn) {
        List<Move> captures = new ArrayList<>();
        for (Move move : generateAllLegalMoves(helper, isWhiteTurn)) {
            if (helper.board[move.to.row][move.to.col] != null) {
                captures.add(move);
            }
        }
        return captures;
    }
    
    private ChessPiece[][] makeMove(ChessPiece[][] board, Move move) {
        ChessPiece[][] newBoard = copyBoard(board);
        newBoard[move.to.row][move.to.col] = newBoard[move.from.row][move.from.col];
        newBoard[move.from.row][move.from.col] = null;
        return newBoard;
    }
    
    private int evaluate(ChessPiece[][] board) {
        int total = 0;
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                ChessPiece piece = board[r][c];
                if (piece != null) {
                    int sign = (piece.isWhite == aiIsWhite) ? 1 : -1;
                    total += sign * getPieceValue(piece, r, c);
                }
            }
        }
        return total;
    }
    
    private int getPieceValue(ChessPiece piece, int row, int col) {
        int value;
        if (piece instanceof Pawn) value = PAWN_VALUE + PAWN_BONUS[row][col];
        else if (piece instanceof Knight) value = KNIGHT_VALUE;
        else if (piece instanceof Bishop) value = BISHOP_VALUE;
        else if (piece instanceof Rook) value = ROOK_VALUE;
        else if (piece instanceof Queen) value = QUEEN_VALUE;
        else if (piece instanceof King) value = KING_VALUE;
        else throw new IllegalArgumentException("Unknown piece");
        
        // Adjust orientation for black pieces
        if (!piece.isWhite) {
            row = 7 - row;
        }
        return value;
    }
    
    private String moveToString(Move move) {
        if (move == null) return "";
        return colToFile(move.from.col) + "" + rowToChessRow(move.from.row) + 
               colToFile(move.to.col) + rowToChessRow(move.to.row);
    }
    
    private static char colToFile(int col) {
        return (char) ('a' + col);
    }
    
    private static int rowToChessRow(int row) {
        return 8 - row;
    }
    
    private static final class Move {
        final Position from;
        final Position to;
        
        Move(Position from, Position to) {
            this.from = from;
            this.to = to;
        }
    }
}