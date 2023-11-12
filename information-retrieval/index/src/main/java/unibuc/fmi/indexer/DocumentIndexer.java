package unibuc.fmi.indexer;

import java.io.IOException;
import java.io.StringReader;
import java.nio.file.Path;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Version;
import org.apache.lucene.analysis.ro.RomanianAnalyzer;

import unibuc.fmi.parse.TikaContent;
import unibuc.fmi.common.Utils;

public class DocumentIndexer implements AutoCloseable {
    private final IndexWriter indexWriter;

    public DocumentIndexer(Path indexPath, Version version) throws IOException {
        // Let Lucene choose the proper algorithm for working with files based on OS
        Directory directory = FSDirectory.open(indexPath);

        // User per-field analyzer
        Analyzer analyzer = new RomanianAnalyzer();

        // Open the index in CREATE mode to override previous segments
        IndexWriterConfig iwConfig = new IndexWriterConfig(analyzer)
                .setOpenMode(IndexWriterConfig.OpenMode.CREATE)
                .setCommitOnClose(true);

        // Open index for writing
        indexWriter = new IndexWriter(directory, iwConfig);
    }

    public void addDoc(TikaContent content) throws IOException {
        Document document = new Document();

        document.add(new TextField("content", new StringReader(content.getText())));

        document.add(new StoredField("filename", content.getFilename()));

        indexWriter.addDocument(document);
    }

    public IndexWriter getIndexWriter() {
        return indexWriter;
    }

    @Override
    public void close() throws Exception {
        indexWriter.close();
    }
}
