package unibuc.fmi;

import org.apache.commons.collections4.map.HashedMap;
import org.apache.lucene.analysis.standard.StandardAnalyzer;

public class Useful {
    public static void Hey() {
        System.out.println("Hey there!");

        HashedMap<String, String> m = new HashedMap<>();
        m.clear();

        // 0. Specify the analyzer for tokenizing text.
        // The same analyzer should be used for indexing and searching
        StandardAnalyzer analyzer = new StandardAnalyzer();

        analyzer.close();
    }
}
