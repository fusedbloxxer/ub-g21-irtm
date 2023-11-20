package unibuc.fmi.query;

import java.nio.file.Path;

import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.search.IndexSearcher;

public final class QueryHandlerFactory {
    private final boolean interactive;
    private final Path[] paths;

    private QueryHandlerFactory(boolean interactive, Path... paths) {
        super();
        this.interactive = interactive;
        this.paths = paths;
    }

    public static QueryHandlerFactory useUserOptions(boolean interactive, Path... paths) {
        return new QueryHandlerFactory(interactive, paths);
    }

    public QueryHandler build(DirectoryReader reader, IndexSearcher searcher, QueryParser parser) {
        if (interactive) {
            return new InteractiveQueryHandler(reader, searcher, parser);
        } else {
            return new FileQueryHandler(reader, searcher, parser, paths);
        }
    }
}
