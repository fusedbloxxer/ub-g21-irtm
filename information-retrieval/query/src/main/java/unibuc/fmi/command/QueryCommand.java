package unibuc.fmi.command;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.AccessDeniedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.Callable;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.ro.RomanianAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.StandardDirectoryReader;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import unibuc.fmi.analyze.RoTextAnalyzer;
import unibuc.fmi.common.Utils;

@Command(name = "query", version = "1.0.0", mixinStandardHelpOptions = true)
public class QueryCommand implements Callable<Integer> {
    @Option(names = { "-i",
            "--index-path" }, paramLabel = "INDEXPATH", defaultValue = ".index", description = "the path from where the index will be read (default: ${DEFAULT-VALUE})")
    Path indexPath;

    @Parameters(index = "0", arity = "1", paramLabel = "QUERY", description = "the query that represents your information need to be found in the indexed documents")
    String queryString;

    @Override
    public Integer call() throws IOException, ParseException {
        if (!Files.exists(indexPath)) {
            throw new FileNotFoundException(indexPath.toAbsolutePath().toString());
        }
        if (!Files.isReadable(indexPath)) {
            throw new AccessDeniedException(indexPath.toAbsolutePath().toString());
        }
        if (!Files.isDirectory(indexPath)) {
            throw new IOException(
                    "File at path does not represent a directory: " + indexPath.toAbsolutePath().toString());
        }

        try (Directory dir = FSDirectory.open(indexPath); DirectoryReader reader = DirectoryReader.open(dir)) {
            Analyzer analyzer = new RoTextAnalyzer(Utils.LUCENE_VERSION);
            IndexSearcher searcher = new IndexSearcher(reader);

            QueryParser queryParser = new QueryParser("content", analyzer);
            Query query = queryParser.parse(queryString);

            TopDocs docs = searcher.search(query, 10);
            StoredFields storedFields = reader.storedFields();
            System.out.println(docs.totalHits + ":" + query.toString());

            for (var scoreDoc : docs.scoreDocs) {
                Document document = storedFields.document(scoreDoc.doc);
                System.out.println(document.getField("filename"));
            }
        }

        return 0;
    }
}
