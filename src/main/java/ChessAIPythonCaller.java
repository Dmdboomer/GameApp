package src.main.java;

    import java.io.BufferedReader;
    import java.io.InputStreamReader;

    public class ChessAIPythonCaller {

        public static String[] getBestMove(String fen) {
            try {
                // 1. Build the command with FEN as argument
                ProcessBuilder pb = new ProcessBuilder("./src/NNBot/.venv/bin/python3", "./src/NNBot/FEN_to_BestMove.py", fen);

                // 2. Start the process
                Process process = pb.start();

                BufferedReader stdOut = new BufferedReader(
                    new InputStreamReader(process.getInputStream())
                );
                BufferedReader stdErr = new BufferedReader(
                    new InputStreamReader(process.getErrorStream())  // ADD ERROR STREAM
                );

                // Read two lines of output: first move and second move
                String move1 = stdOut.readLine();
                String move2 = stdOut.readLine();

                // NEW: Capture error messages
                StringBuilder errorOutput = new StringBuilder();
                String errLine;
                while ((errLine = stdErr.readLine()) != null) {
                    errorOutput.append(errLine).append("\n");
                }

                int exitCode = process.waitFor();

                if (exitCode != 0) {
                    throw new RuntimeException(
                        "Python error [" + exitCode + "]: " + errorOutput.toString()
                    );
                }

                // Return an array of two moves, converting "null" to null
                return new String[] { 
                    "null".equals(move1) ? null : move1, 
                    "null".equals(move2) ? null : move2 
                };

            } catch (Exception e) {
                throw new RuntimeException("Error calling Python script", e);
            }
        }

        // Example usage
        public static void main(String[] args) {
            String fen = "rnbq1b1r/1pppPk2/6pp/p2P4/5B2/8/PPP2PPP/RN1QKBNR b KQ - 0 0";
            String[] bestMoves = getBestMove(fen);
            System.out.println("Model1: " + bestMoves[0]);
            System.out.println("Model2: " + bestMoves[1]);
        }
    }