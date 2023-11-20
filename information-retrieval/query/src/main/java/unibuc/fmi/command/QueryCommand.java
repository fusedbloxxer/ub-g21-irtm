package unibuc.fmi.command;

import java.nio.file.AccessDeniedException;
import java.io.FileNotFoundException;
import java.util.concurrent.Callable;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.Directory;

import picocli.CommandLine.ArgGroup;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import unibuc.fmi.analyzer.RoTextAnalyzer;
import unibuc.fmi.analyzer.AnalyzerMode;
import unibuc.fmi.analyzer.AnalyzerOptions;
import unibuc.fmi.common.Utils;
import unibuc.fmi.query.QueryHandlerFactory;

@Command(name = "query", version = "1.0.0", mixinStandardHelpOptions = true)
public class QueryCommand implements Callable<Integer> {
    public static class ExclusiveArgs {
        @Option(names = {
                "--it",
                "--interactive" }, required = true, description = "specify queries interactively (default: ${DEFAULT-VALUE})")
        boolean interactive;

        @Option(names = { "-q",
                "--query-path" }, arity = "1..*", paramLabel = "QUERYPATH", defaultValue = "queries.txt", required = true, description = "specify queries from a list of files (default: ${DEFAULT-VALUE})")
        Path[] paths;
    }

    @ArgGroup(exclusive = true, multiplicity = "1")
    ExclusiveArgs queryArgs;

    @Option(names = { "-i",
            "--index-path" }, paramLabel = "INDEXPATH", defaultValue = ".index", description = "the path from where the index will be read (default: ${DEFAULT-VALUE})")
    Path indexPath;

    @Option(names = { "--debug" }, negatable = true, description = "enable debugging mode (default: ${DEFAULT-VALUE})")
    boolean debug = false;

    @Override
    public Integer call() throws IOException, ParseException {
        // Leverage CLI options to setup execution
        setup();

        // Validate arguments
        validateArguments();

        // Process the user queries
        processQueries();

        // Return success
        return 0;
    }

    private void setup() {
        Utils.IsDebug = debug;
    }

    private void validateArguments() throws IOException {
        if (!Files.exists(indexPath)) {
            throw new FileNotFoundException(indexPath.toAbsolutePath().toString());
        }
        if (!Files.isReadable(indexPath)) {
            throw new AccessDeniedException(indexPath.toAbsolutePath().toString());
        }
        if (!Files.isDirectory(indexPath)) {
            throw new IOException("File at path does not represent a directory: " + indexPath.toAbsolutePath());
        }
    }

    private int processQueries() throws IOException, ParseException {
        // Open the directory of the index
        try (Directory dir = FSDirectory.open(indexPath); DirectoryReader reader = DirectoryReader.open(dir)) {
            // Create analyzer in query mode and prepare for searching
            AnalyzerOptions options = new AnalyzerOptions(AnalyzerMode.QUERYING, false);
            Analyzer analyzer = new RoTextAnalyzer(options);
            QueryParser parser = new QueryParser("content", analyzer);
            IndexSearcher searcher = new IndexSearcher(reader);

            // Handle the given input in different ways according to the user options
            QueryHandlerFactory
                    .useUserOptions(queryArgs.interactive, queryArgs.paths)
                    .build(reader, searcher, parser)
                    .handle();

            // Return success
            return 0;
        }
    }
}
