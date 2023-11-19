package unibuc.fmi.queryhandler;

import org.apache.lucene.queryparser.classic.QueryParser;

import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.SubmissionPublisher;
import java.util.concurrent.CyclicBarrier;

import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.search.IndexSearcher;

public class InteractiveQueryHandler implements QueryHandler {
    @SuppressWarnings("unused")
    private final SimpleQueryHandler delegate;
    private final SubmissionPublisher<String> publisher;
    private final CyclicBarrier barrier;

    public InteractiveQueryHandler(DirectoryReader reader, IndexSearcher searcher, QueryParser parser) {
        this.barrier = new CyclicBarrier(2);
        this.publisher = new SubmissionPublisher<>();
        this.delegate = new SimpleQueryHandler(reader, searcher, parser, publisher, barrier);
    }

    @Override
    public void handle() {
        try {
            while (true) {
                String queryString = System.console().readLine("Enter query: ");
                this.publisher.submit(queryString);
                this.barrier.await();
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (BrokenBarrierException e) {
            e.printStackTrace();
        } finally {
            this.publisher.close();
        }
    }
}
