package unibuc.fmi.command;

import java.nio.file.Path;
import java.util.concurrent.Callable;

import unibuc.fmi.common.Utils;
import picocli.CommandLine.Option;
import unibuc.fmi.file.PathSearch;
import picocli.CommandLine.Command;
import unibuc.fmi.parse.TikaParser;
import picocli.CommandLine.Parameters;
import unibuc.fmi.file.TikaDocTypeFilter;
import unibuc.fmi.indexer.DocumentIndexer;
import unibuc.fmi.document.DocumentFields;
import unibuc.fmi.file.PathSearch.PathSearchOptions;

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
        var analyzer = docIndexer.getIndexWriter().getAnalyzer();
        var docFilter = new TikaDocTypeFilter();
        var tika = new TikaParser();

        // Save files into the Index
        for (Path filepath : filepaths) {
            // Parse document
            var content = tika.parse(filepath);

            // Skip document with errors
            if (content.isEmpty()) {
                continue;
            }

            // Skip documents with non-supported type
            if (!docFilter.accept(content.get().getMediaType())) {
                if (Utils.IsDebug) {
                    System.out.println("[Debug] File is ignored: " + content.get().getFilepath());
                }
            }

            // Inspect the process
            if (Utils.IsDebug) {
                System.out.println("[Debug] Adding document from: " + content.get().getFilepath());
                System.out.println("[Debug] MEDIA_TYPE: " + content.get().getMediaType());
                Utils.debugAnalyzer(analyzer, DocumentFields.FIELD_CONTENT, content.get().getText());
            }

            // Index content
            docIndexer.addDoc(content.get());
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
