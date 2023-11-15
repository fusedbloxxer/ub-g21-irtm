package unibuc.fmi.command;

import java.nio.file.Path;
import java.util.concurrent.Callable;

import org.apache.lucene.document.TextField;

import picocli.CommandLine.Parameters;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import unibuc.fmi.file.PathSearch;
import unibuc.fmi.file.PathSearch.PathSearchOptions;
import unibuc.fmi.indexer.DocumentIndexer;
import unibuc.fmi.parse.TikaParser;
import unibuc.fmi.common.Utils;
import unibuc.fmi.document.DocumentFields;

@Command(name = "index", version = "1.0.0", mixinStandardHelpOptions = true, exitCodeList = {})
public class IndexCommand implements Callable<Integer> {
    @Parameters(index = "0", arity = "1", paramLabel = "DATAPATH", description = "the path to a regular or directory file")
    Path dataPath;

    @Option(names = { "-i",
            "--index-path" }, paramLabel = "INDEXPATH", defaultValue = ".index", description = "the path to where the index will be saved (default: ${DEFAULT-VALUE})")
    Path indexPath;

    @Option(names = { "-f",
            "--file-type" }, paramLabel = "FILETYPE", split = ",", description = "supported filetype formats")
    String[] supportedFileTypes = new String[] {};

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
            Utils.debugAnalyzer(docIndexer.getIndexWriter().getAnalyzer(), DocumentFields.FIELD_CONTENT,
                    content.getText());

            // Index content
            docIndexer.addDoc(content);
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
