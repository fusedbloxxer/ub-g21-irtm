package unibuc.fmi.analyzer;

import org.apache.lucene.analysis.miscellaneous.ASCIIFoldingFilter;
import org.apache.lucene.analysis.miscellaneous.DropIfFlaggedFilter;
import org.apache.lucene.analysis.pattern.PatternCaptureGroupTokenFilter;
import org.apache.lucene.analysis.pattern.PatternTypingFilter;
import org.apache.lucene.analysis.pattern.PatternTypingFilter.PatternTypingRule;
import org.apache.lucene.analysis.snowball.SnowballFilter;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.synonym.SynonymGraphFilter;
import org.tartarus.snowball.ext.RomanianStemmer;

import java.util.regex.Pattern;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.StopFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.core.FlattenGraphFilter;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;

import unibuc.fmi.attributes.TokenFlagsAttribute.TokenFlag;
import unibuc.fmi.filters.ConditionalTokenFlagsFilter;
import unibuc.fmi.filters.PatternTokenFlagsFilter;

public class RoTextAnalyzer extends Analyzer {
    private final AnalyzerOptions options;

    public RoTextAnalyzer(AnalyzerOptions options) {
        super();
        this.options = options;
    }

    @Override
    public TokenStreamComponents createComponents(String fieldName) {
        Tokenizer tz;
        TokenStream ts;

        // Split by whitespace
        ts = tz = new StandardTokenizer();

        // Insert token flags representing the type
        ts = new PatternTokenFlagsFilter(ts);

        // Remove tokens made solely out of punctuation
        ts = new PatternTypingFilter(ts,
                new PatternTypingRule(Pattern.compile("^\\p{Punct}+$"), AnalyzerFlags.FLAG_PUNCTUATION, ""));
        ts = new DropIfFlaggedFilter(ts, AnalyzerFlags.FLAG_PUNCTUATION);

        // Emit subtokens for various token types
        ts = new ConditionalTokenFlagsFilter(ts, is -> new PatternCaptureGroupTokenFilter(is, true, new Pattern[] {
                Pattern.compile("(?:https?://)?([a-zA-Z-_0-9]+)")
        }), true, TokenFlag.UrlAddress);
        ts = new ConditionalTokenFlagsFilter(ts, is -> new PatternCaptureGroupTokenFilter(is, true, new Pattern[] {
                Pattern.compile("([a-zA-Z0-9_.+-]+)@"),
                Pattern.compile("([a-zA-Z-_0-9]+|[a-zA-Z-_0-9]+\\.?)+")
        }), true, TokenFlag.EmailAddress);

        // Transform all tokens to lowercase
        ts = new LowerCaseFilter(ts);

        // Insert synonyms at query time
        if (options.isIndexing() && options.usesSynonyms()) {
            var synonyms = AnalyzerResources.loadSynonyms(true);
            if (synonyms.isPresent()) {
                ts = new SynonymGraphFilter(ts, synonyms.get(), true);
                ts = new FlattenGraphFilter(ts);
            }
        }

        // Remove stopwords with accents
        var stopwordsAccents = AnalyzerResources.loadStopwords(true);
        if (stopwordsAccents.isPresent()) {
            ts = new ConditionalTokenFlagsFilter(ts,
                    is -> new StopFilter(is, stopwordsAccents.get()), true, TokenFlag.Word);
        }

        // Apply stemming
        ts = new SnowballFilter(ts, new RomanianStemmer());

        // Remove diacritics
        ts = new ConditionalTokenFlagsFilter(ts,
                is -> new ASCIIFoldingFilter(is), true, TokenFlag.Word);

        // Remove stopwords without accents
        var stopwordsNoAccents = AnalyzerResources.loadStopwords(false);
        if (stopwordsNoAccents.isPresent()) {
            ts = new ConditionalTokenFlagsFilter(ts,
                    is -> new StopFilter(is, stopwordsNoAccents.get()), true, TokenFlag.Word);
        }

        // Combine all components
        return new TokenStreamComponents(tz, ts);
    }
}
