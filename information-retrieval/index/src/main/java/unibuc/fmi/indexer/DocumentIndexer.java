package unibuc.fmi.indexer;

import java.io.IOException;
import java.nio.file.Path;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.Version;

import unibuc.fmi.analyze.RoTextAnalyzer;
import unibuc.fmi.common.Constants;
import unibuc.fmi.parse.TikaContent;

public class DocumentIndexer implements AutoCloseable {
    private final IndexWriter indexWriter;

    public DocumentIndexer(Path indexPath, Version version) throws IOException {
        // Let Lucene choose the proper algorithm for working with files based on OS
        Directory directory = Constants.IsDebug ? new ByteBuffersDirectory() : FSDirectory.open(indexPath);

        // Custom analyzer for romanian text
        // Analyzer analyzer = new RoTextAnalyzer(version);
        Analyzer analyzer = new StandardAnalyzer();

        // Open the index in CREATE mode to override previous segments
        IndexWriterConfig iwConfig = new IndexWriterConfig(analyzer)
                .setOpenMode(IndexWriterConfig.OpenMode.CREATE)
                .setCommitOnClose(true);

        // Open index for writing
        indexWriter = new IndexWriter(directory, iwConfig);
    }

    public void addDoc(TikaContent content) throws IOException {
        Document document = new Document();
        document.add(new TextField("content", content.getReader()));
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
