package unibuc.fmi.query;

import java.util.List;
import java.util.Arrays;
import java.io.IOException;
import java.nio.file.FileVisitOption;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.SubmissionPublisher;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;

public class FileQueryHandler implements QueryHandler {
    @SuppressWarnings("unused")
    private final SimpleQueryHandler delegate;
    private final SubmissionPublisher<String> publisher;
    private final CyclicBarrier barrier;
    private final List<Path> paths;

    public FileQueryHandler(DirectoryReader reader, IndexSearcher searcher, QueryParser parser, Path... paths) {
        this.barrier = new CyclicBarrier(2);
        this.paths = Arrays.asList(paths);
        this.publisher = new SubmissionPublisher<>();
        this.delegate = new SimpleQueryHandler(reader, searcher, parser, publisher, barrier);
    }

    @Override
    public void handle() {
        paths
                .stream()
                .flatMap(this::searchTree)
                .filter(Files::isRegularFile)
                .filter(Files::isReadable)
                .map(this::fileContent)
                .filter(x -> x != null && x.length() != 0)
                .flatMap(content -> Stream.of(content.split(System.lineSeparator())))
                .filter(x -> x.length() != 0)
                .forEach(this::sendQuery);
        this.publisher.close();
    }

    private Stream<Path> searchTree(Path path) {
        try (Stream<Path> subPath = Files.walk(path, new FileVisitOption[] {})) {
            return subPath.collect(Collectors.toList()).stream();
        } catch (IOException e) {
            System.err.println("Could not read path: " + path + ". " + e.getMessage());
            e.printStackTrace();
            return Stream.empty();
        }
    }

    private String fileContent(Path path) {
        try {
            System.out.println("Query File: " + path);
            return Files.readString(path);
        } catch (IOException e) {
            System.err.println("Could not read query file: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    private void sendQuery(String query) {
        try {
            this.publisher.submit(query);
            this.barrier.await();
        } catch (InterruptedException | BrokenBarrierException e) {
            sendQuery("Barrier synchronization error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
