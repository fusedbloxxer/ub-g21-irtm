package unibuc.fmi.common;

import java.io.IOException;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.util.Version;

public class Utils {
    // Constants
    public static final Version LUCENE_VERSION = Version.LATEST;

    // Flags
    public static boolean IsDebug = false;

    // DebuggingUtilities
    public static void debugAnalyzer(Analyzer analyzer, String name, String value) {
        if (!Utils.IsDebug) {
            return;
        }

        System.out.println("[Debug] Inspecting analysis...");
        System.out.println("[Debug] FieldName: " + name);
        try (TokenStream ts = analyzer.tokenStream(name, value)) {
            // Required to reset the token stream!
            ts.reset();

            // Extract info from each token
            OffsetAttribute offAttr = ts.addAttribute(OffsetAttribute.class);
            CharTermAttribute termAttr = ts.addAttribute(CharTermAttribute.class);
            PositionIncrementAttribute posAttr = ts.addAttribute(PositionIncrementAttribute.class);
            System.out.print("[Debug] ");

            while (ts.incrementToken()) {
                System.out.printf("[%s, (%d, %d), %d] ", termAttr.toString(), offAttr.startOffset(),
                        offAttr.endOffset(),
                        posAttr.getPositionIncrement());
            }
            System.out.println();
            System.out.println("[Debug] Analysis complete.");

            ts.end();
        } catch (IOException e) {
            System.err.println("[Debug] Token analysis failed");
        }
    }
}
