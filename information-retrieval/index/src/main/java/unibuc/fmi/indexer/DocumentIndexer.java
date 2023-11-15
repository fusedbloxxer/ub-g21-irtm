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

import unibuc.fmi.analyze.RoTextAnalyzer;
import unibuc.fmi.document.DocumentFields;
import unibuc.fmi.parse.TikaContent;

public class DocumentIndexer implements AutoCloseable {
    private final IndexWriter indexWriter;

    public DocumentIndexer(Path indexPath, Version version) throws IOException {
        // Let Lucene choose the proper algorithm for working with files based on OS
        Directory directory = FSDirectory.open(indexPath);

        // User per-field analyzer
        Analyzer analyzer = new RoTextAnalyzer(version);

        // Open the index in CREATE mode to override previous segments
        IndexWriterConfig iwConfig = new IndexWriterConfig(analyzer)
                .setOpenMode(IndexWriterConfig.OpenMode.CREATE)
                .setCommitOnClose(true);

        // Open index for writing
        indexWriter = new IndexWriter(directory, iwConfig);
    }

    public void addDoc(TikaContent content) throws IOException {
        // Creat a new document to add
        Document document = new Document();

        // Add fields
        document.add(new TextField(DocumentFields.FIELD_CONTENT, new StringReader(content.getText())));
        document.add(new StoredField(DocumentFields.FIELD_FILENAME, content.getFilename()));

        // Add document with populated fields
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