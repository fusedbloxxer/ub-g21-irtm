package unibuc.fmi.common;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.io.FileWriter;
import java.nio.file.Path;
import java.util.Optional;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;

import unibuc.fmi.attributes.TokenFlagsAttribute;

public class Utils {
    // Constants
    public static final Version LUCENE_VERSION = Version.LATEST;

    // Flags
    public static boolean IsDebug = false;

    // DebuggingUtilities
    public static final Path DEBUG_OUTPUT_PATH = Path.of(".debug/output.txt");

    // Print to the console the extracted tokens and also save them to a file.
    public static void debugAnalyzer(Analyzer analyzer, String name, String value) {
        if (!Utils.IsDebug) { // Do not run if debug mode is disabled.
            return;
        }

        // Save tokens to a file to ease debugging in case the outpu is too big.
        Optional<PrintWriter> debugOutputWriter = Optional.empty();
        try {
            if (!Files.exists(DEBUG_OUTPUT_PATH)) {
                Files.createDirectories(DEBUG_OUTPUT_PATH.getParent());
                Files.createFile(DEBUG_OUTPUT_PATH);
            }
            debugOutputWriter = Optional
                    .of(new PrintWriter(new BufferedWriter(new FileWriter(DEBUG_OUTPUT_PATH.toFile()))));
        } catch (IOException e) {
            System.err.println("[Debug] Cannot write tokens to debug output file: " + e.getMessage());
        }

        // Analyze explicitly and record each token both in console and in the file.
        System.out.println("[Debug] Inspecting analysis...");
        System.out.println("[Debug] FieldName: " + name);
        try (TokenStream ts = analyzer.tokenStream(name, value)) {
            // Required to reset the token stream!
            ts.reset();

            // Extract info from each token
            TokenFlagsAttribute tokenFlagAttr = ts.addAttribute(TokenFlagsAttribute.class);
            CharTermAttribute termAttr = ts.addAttribute(CharTermAttribute.class);
            System.out.print("[Debug] ");

            while (ts.incrementToken()) {
                System.out.printf("[%s]: %s %s\n", termAttr.toString(), tokenFlagAttr.getTokenFlags(),
                        tokenFlagAttr.isFinalToken());
                debugOutputWriter
                        .ifPresent(x -> x.printf("[%s]: %s\n", termAttr.toString(), tokenFlagAttr.getTokenFlags()));
            }

            System.out.println();
            System.out.println("[Debug] Analysis complete.");
            ts.end();
        } catch (IOException e) {
            System.err.println("[Debug] Token analysis failed: " + e.getMessage());
        } finally {
            debugOutputWriter.ifPresent(x -> x.close());
        }
    }
}
