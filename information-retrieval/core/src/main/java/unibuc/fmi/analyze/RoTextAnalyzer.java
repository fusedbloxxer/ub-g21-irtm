package unibuc.fmi.analyze;

import org.apache.lucene.analysis.standard.StandardTokenizer;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.util.Version;

public class RoTextAnalyzer extends Analyzer {
    public RoTextAnalyzer(Version version) {
    }

    @Override
    public TokenStreamComponents createComponents(String fieldName) {
        Tokenizer start = new StandardTokenizer();
        TokenStream middle = new LowerCaseFilter(start);
        return new TokenStreamComponents(start, middle);
    }
}
