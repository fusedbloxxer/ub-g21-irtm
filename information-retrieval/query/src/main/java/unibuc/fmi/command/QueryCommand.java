package unibuc.fmi.command;

import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.Callable;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import picocli.CommandLine.Command;
import picocli.CommandLine.Parameters;
import unibuc.fmi.analyze.RoTextAnalyzer;
import unibuc.fmi.common.Utils;

@Command(name = "query", version = "1.0.0", mixinStandardHelpOptions = true)
public class QueryCommand implements Callable<Integer> {
    @Parameters(index = "0", arity = "0..1", paramLabel = "INDEXPATH", defaultValue = ".index", description = "the path from where the index will be read (default: ${DEFAULT-VALUE})")
    Path indexPath;

    @Parameters(index = "1", arity = "1", paramLabel = "QUERY", description = "the query that represents your information need to be found in the indexed documents")
    String queryString;

    @Override
    public Integer call() throws IOException, ParseException {
        try (Directory directory = FSDirectory.open(indexPath);
                IndexReader indexReader = DirectoryReader.open(directory)) {
            IndexSearcher searcher = new IndexSearcher(indexReader);
            Analyzer analyzer = new StandardAnalyzer();

            QueryParser queryParser = new QueryParser("content", analyzer);
            Query query = queryParser.parse(queryString);

            TopDocs docs = searcher.search(query, 10);
            StoredFields storedFields = indexReader.storedFields();
            System.out.println(docs.totalHits + ":" + query.toString());

            for (var scoreDoc : docs.scoreDocs) {
                Document document = storedFields.document(scoreDoc.doc);
                System.out.println(document.getField("filename"));
            }
        }

        return 0;
    }
}
