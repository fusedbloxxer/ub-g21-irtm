package unibuc.fmi.analyze;

import org.apache.lucene.analysis.miscellaneous.ASCIIFoldingFilter;
import org.apache.lucene.analysis.miscellaneous.DropIfFlaggedFilter;
import org.apache.lucene.analysis.pattern.PatternCaptureGroupTokenFilter;
import org.apache.lucene.analysis.pattern.PatternTypingFilter;
import org.apache.lucene.analysis.pattern.PatternTypingFilter.PatternTypingRule;
import org.apache.lucene.analysis.snowball.SnowballFilter;
import org.apache.lucene.analysis.standard.StandardTokenizer;
import org.apache.lucene.analysis.synonym.SynonymGraphFilter;
import org.apache.lucene.analysis.synonym.SynonymMap;
import org.apache.lucene.util.CharsRef;
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
import org.apache.lucene.analysis.core.FlattenGraphFilter;

import unibuc.fmi.analyze.attributes.TokenFlagsAttribute.TokenFlag;
import unibuc.fmi.analyze.filters.ConditionalTokenFlagsFilter;
import unibuc.fmi.analyze.filters.PatternTokenFlagsFilter;

public class RoTextAnalyzer extends Analyzer {
    public enum AnalyzerMode {
        INDEXING,
        QUERYING,
    }

    private static final Path PATH_RES = Path.of(".", "resources");
    private static final Path PATH_SYN_SINGL = PATH_RES.resolve("synonyms_singl.txt");
    private static final Path PATH_SYN_MULTI = PATH_RES.resolve("synonyms_multi.txt");
    private static final Path PATH_STOPWORDS_FILE = PATH_RES.resolve("stopwords.txt");
    private static final Path PATH_STOPWORDS_NO_ACCENTS_FILE = PATH_RES.resolve("stopwords_no_accents.txt");
    private static final int FLAG_PUNCTUATION = 1;
    private final boolean allowSynonyms = false;
    private final AnalyzerMode mode;

    public RoTextAnalyzer(AnalyzerMode mode) {
        super();
        this.mode = mode;
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

        // Insert synonyms at query time
        if (mode == AnalyzerMode.INDEXING && allowSynonyms) {
            var synonyms = loadSynonyms(true);
            if (synonyms.isPresent()) {
                ts = new SynonymGraphFilter(ts, synonyms.get(), true);
                ts = new FlattenGraphFilter(ts);
            }
        }

        // Remove stopwords with accents
        var stopwordsAccents = loadStopwords(true);
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
        var stopwordsNoAccents = loadStopwords(false);
        if (stopwordsNoAccents.isPresent()) {
            ts = new ConditionalTokenFlagsFilter(ts,
                    is -> new StopFilter(is, stopwordsNoAccents.get()), true, TokenFlag.Word);
        }

        // Combine all components
        return new TokenStreamComponents(tz, ts);
    }

    private Optional<CharArraySet> loadStopwords(boolean accents) {
        Path path = accents ? PATH_STOPWORDS_FILE : PATH_STOPWORDS_NO_ACCENTS_FILE;
        try (var reader = new BufferedReader(new FileReader(path.toFile()))) {
            return Optional.of(WordlistLoader.getWordSet(reader));
        } catch (IOException e) {
            System.err.println("Could not read stopwords file: " + e.getMessage());
            return Optional.empty();
        }
    }

    private Optional<SynonymMap> loadSynonyms(boolean multi) {
        Path path = multi ? PATH_SYN_MULTI : PATH_SYN_SINGL;
        try (BufferedReader reader = new BufferedReader(new FileReader(path.toFile()))) {
            SynonymMap.Builder builder = new SynonymMap.Builder();
            for (String l = reader.readLine(); l != null; l = reader.readLine()) {
                String[] pair = l.split("\s");
                builder.add(new CharsRef(pair[0]), new CharsRef(pair[1]), true);
            }
            return Optional.of(builder.build());
        } catch (IOException e) {
            System.err.println("Could not read synonyms file: " + e.getMessage());
            return Optional.empty();
        }
    }
}
