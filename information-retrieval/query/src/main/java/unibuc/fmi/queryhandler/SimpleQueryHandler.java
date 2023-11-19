package unibuc.fmi.queryhandler;

import java.io.IOException;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.Flow.Publisher;
import java.util.concurrent.Flow.Subscriber;
import java.util.concurrent.Flow.Subscription;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;

import unibuc.fmi.document.DocumentFields;

public class SimpleQueryHandler implements Subscriber<String> {
    private final DirectoryReader reader;
    private final IndexSearcher searcher;
    private final CyclicBarrier barrier;
    private final QueryParser parser;
    private Subscription subscription;

    public SimpleQueryHandler(DirectoryReader reader, IndexSearcher searcher, QueryParser parser,
            Publisher<String> publisher, CyclicBarrier barrier) {
        this.reader = reader;
        this.parser = parser;
        this.barrier = barrier;
        this.searcher = searcher;
        publisher.subscribe(this);
    }

    @Override
    public void onSubscribe(Subscription subscription) {
        this.subscription = subscription;
        this.subscription.request(1);
    }

    @Override
    public void onNext(String queryString) {
        try {
            // Inform the user with regards to the current query
            System.out.println("Raw Query: " + queryString);

            // Parse the query and search the index
            Query query = parser.parse(queryString);
            TopDocs docs = searcher.search(query, 10);
            StoredFields fields = reader.storedFields();

            // See how many results we obtained.
            System.out.println("Parsed Query: " + query.toString());
            System.out.println("Found: " + docs.totalHits);

            // Show ordered results
            AtomicInteger counter = new AtomicInteger(0);

            // Retrieve each found document
            Stream
                    .of(docs.scoreDocs)
                    .forEach(scoreDoc -> {
                        try {
                            Document document = fields.document(scoreDoc.doc);
                            System.out.print(counter.addAndGet(1) + ". ");
                            System.out.println(document.getField(DocumentFields.FIELD_FILEPATH).stringValue());
                        } catch (IOException e) {
                            System.err.println("Could not retrieve found doc: " + e.getMessage());
                            e.printStackTrace();
                        }
                    });

            // Newline as delimiter between queries
            System.out.println();
        } catch (ParseException | IOException e) {
            System.err.println("Query unknown error: " + e.getMessage());
            e.printStackTrace();
        } finally {
            try {
                this.subscription.request(1);
                this.barrier.await();
            } catch (InterruptedException | BrokenBarrierException e) {
                System.err.println("Synchronization error with the barrier: " + e.getMessage());
                e.printStackTrace();
            }
        }
    }

    @Override
    public void onComplete() {
    }

    @Override
    public void onError(Throwable t) {
        t.printStackTrace();
    }

    public Subscription geSubscription() {
        return this.subscription;
    }
}
