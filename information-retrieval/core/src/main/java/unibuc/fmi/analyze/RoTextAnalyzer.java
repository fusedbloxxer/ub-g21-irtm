package unibuc.fmi.analyze;

import org.apache.lucene.analysis.miscellaneous.ASCIIFoldingFilter;
import org.apache.lucene.analysis.miscellaneous.DropIfFlaggedFilter;
import org.apache.lucene.analysis.pattern.PatternCaptureGroupTokenFilter;
import org.apache.lucene.analysis.pattern.PatternTypingFilter;
import org.apache.lucene.analysis.pattern.PatternTypingFilter.PatternTypingRule;
import org.apache.lucene.analysis.snowball.SnowballFilter;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.tartarus.snowball.ext.RomanianStemmer;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Optional;
import java.util.regex.Pattern;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.StopFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.WordlistLoader;

import unibuc.fmi.analyze.attributes.TokenFlagsAttribute.TokenFlag;
import unibuc.fmi.analyze.filters.ConditionalTokenFlagsFilter;
import unibuc.fmi.analyze.filters.PatternTokenFlagsFilter;

public class RoTextAnalyzer extends Analyzer {
    public enum AnalyzerMode {
        INDEXING,
        QUERYING,
    }

    private static final Path PATH_STOPWORDS_NO_ACCENTS_FILE = Path.of(".", "resources", "stopwords_no_accents.txt");
    private static final Path PATH_STOPWORDS_FILE = Path.of(".", "resources", "stopwords.txt");
    private static final int FLAG_PUNCTUATION = 1;

    public RoTextAnalyzer(AnalyzerMode mode) {
        super();
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
        ts = new PatternTypingFilter(ts, new PatternTypingRule(Pattern.compile("^\\p{Punct}+$"), FLAG_PUNCTUATION, ""));
        ts = new DropIfFlaggedFilter(ts, FLAG_PUNCTUATION);

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

        // Remove stopwords with accents
        var stopwordsAccents = buildStopwords(true);
        if (stopwordsAccents.isPresent()) {
            ts = new ConditionalTokenFlagsFilter(ts,
                    is -> new StopFilter(is, stopwordsAccents.get()), true, TokenFlag.Word);
        }

        // Apply stemming
        ts = new ConditionalTokenFlagsFilter(ts,
                is -> new SnowballFilter(is, new RomanianStemmer()), true, TokenFlag.Word);

        // Remove diacritics
        ts = new ConditionalTokenFlagsFilter(ts,
                is -> new ASCIIFoldingFilter(is), true, TokenFlag.Word);

        // Remove stopwords without accents
        var stopwordsNoAccents = buildStopwords(false);
        if (stopwordsNoAccents.isPresent()) {
            ts = new ConditionalTokenFlagsFilter(ts,
                    is -> new StopFilter(is, stopwordsNoAccents.get()), true, TokenFlag.Word);
        }

        // Combine all components
        return new TokenStreamComponents(tz, ts);
    }

    private Optional<CharArraySet> buildStopwords(boolean accents) {
        Path stopwordsPath = accents ? PATH_STOPWORDS_FILE : PATH_STOPWORDS_NO_ACCENTS_FILE;
        try (var reader = new BufferedReader(new FileReader(stopwordsPath.toFile()))) {
            return Optional.of(WordlistLoader.getWordSet(reader));
        } catch (IOException e) {
            System.err.println("Could not read stopwords file: " + e.getMessage());
            return Optional.empty();
        }
    }
}
