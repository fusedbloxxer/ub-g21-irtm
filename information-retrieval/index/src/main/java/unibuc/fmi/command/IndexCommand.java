package unibuc.fmi.command;

import java.nio.file.Path;
import java.util.concurrent.Callable;

import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;

import picocli.CommandLine.Parameters;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import unibuc.fmi.file.PathSearch;
import unibuc.fmi.file.PathSearch.PathSearchOptions;
import unibuc.fmi.indexer.DocumentIndexer;
import unibuc.fmi.parse.TikaParser;
import unibuc.fmi.common.Utils;

@Command(name = "index", version = "1.0.0", mixinStandardHelpOptions = true, exitCodeList = {})
public class IndexCommand implements Callable<Integer> {
    @Parameters(index = "0", arity = "1", paramLabel = "DATAPATH", description = "the path to a regular or directory file")
    Path dataPath;

    @Parameters(index = "1", arity = "0..1", paramLabel = "INDEXPATH", defaultValue = ".index", description = "the path to where the index will be saved (default: ${DEFAULT-VALUE})")
    Path indexPath;

    @Option(names = "--file-type", paramLabel = "FILETYPE", defaultValue = "txt,pdf,docx", split = ",", description = "supported filetype formats (default: ${DEFAULT-VALUE})")
    String[] supportedFileTypes;

    @Option(names = { "--debug" }, negatable = true, description = "enable debugging mode (default: ${DEFAULT-VALUE})")
    boolean debug = false;

    public Integer call() throws Exception {
        // Leverage CLI options to setup execution
        setup();

        // Obtain the paths to the files of interest recursively
        var filepaths = PathSearch.from(dataPath, new PathSearchOptions(supportedFileTypes));

        // Create utils
        var docIndexer = new DocumentIndexer(indexPath, Utils.LUCENE_VERSION);
        var tika = new TikaParser();

        // Save files into the Index
        for (Path filepath : filepaths) {
            // Parse document
            var raw = tika.parse(filepath);

            // Skip document with errors
            if (raw.isEmpty()) {
                continue;
            }

            // Show content
            var content = raw.get();
            System.out.println("FILENAME: " + content.getFilename());
            System.out.println("MEDIA_TYPE: " + content.getMediaType());

            // Index content
            docIndexer.addDoc(content);

            // Debug
            TokenStream ts = docIndexer
                    .getIndexWriter()
                    .getAnalyzer()
                    .tokenStream("content", content.getText());
            CharTermAttribute termAtt = ts.addAttribute(CharTermAttribute.class);
            System.out.println("TOKENS:");

            // Iterate and show terms
            ts.reset();
            while (ts.incrementToken()) {
                System.out.print("[" + termAtt.toString() + "]");
            }
            System.out.println();
            ts.end();
            ts.close();
            System.out.println("------------------------------");
        }

        // Commit the changes
        docIndexer.getIndexWriter().commit();

        // Close the indexer
        docIndexer.close();
        return 0;
    }

    private void setup() {
        // Optionally enable inspection mode
        Utils.IsDebug = debug;
    }
}
