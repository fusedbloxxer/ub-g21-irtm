package unibuc.fmi.analyzer;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Optional;

import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.WordlistLoader;
import org.apache.lucene.analysis.synonym.SynonymMap;
import org.apache.lucene.util.CharsRef;

public class AnalyzerResources {
    public static final Path PATH_RES = Path.of(".", "resources");
    public static final Path PATH_SYN_SINGL = PATH_RES.resolve("synonyms_singl.txt");
    public static final Path PATH_SYN_MULTI = PATH_RES.resolve("synonyms_multi.txt");
    public static final Path PATH_STOPWORDS_FILE = PATH_RES.resolve("stopwords.txt");
    public static final Path PATH_STOPWORDS_NO_ACCENTS_FILE = PATH_RES.resolve("stopwords_no_accents.txt");

    /**
     * Load a set of stopwords from a file stored on disk.
     *
     * @param accents Determines whether the stopwords will have their accents
     *                removed or not.
     * @return A set of stopwords or empty if errors took place.
     */
    public static Optional<CharArraySet> loadStopwords(boolean accents) {
        Path path = accents ? PATH_STOPWORDS_FILE : PATH_STOPWORDS_NO_ACCENTS_FILE;
        try (var reader = new BufferedReader(new FileReader(path.toFile()))) {
            return Optional.of(WordlistLoader.getWordSet(reader));
        } catch (IOException e) {
            System.err.println("Could not read stopwords file: " + e.getMessage());
            return Optional.empty();
        }
    }

    /**
     * Load paired synonyms from disk where they are separated by space and newline;
     * The multi-token synonyms should be separated by 0 ASCII char;
     * All pairs, N * (N - 1), are provided in the default files.
     *
     * @param multiToken If true, allow multi-token synonyms. At indexing time this
     *                   might cause loss of information due to positionIncrements.
     * @return A synonym map or empty if errors took place.
     */
    public static Optional<SynonymMap> loadSynonyms(boolean multiToken) {
        Path path = multiToken ? PATH_SYN_MULTI : PATH_SYN_SINGL;
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
