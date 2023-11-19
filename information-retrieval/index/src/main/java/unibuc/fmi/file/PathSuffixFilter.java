package unibuc.fmi.file;

import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.nio.file.PathMatcher;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class PathSuffixFilter implements Filter<Path> {
    private final PathMatcher extMatcher;

    public PathSuffixFilter(String... fileExtensions) {
        extMatcher = fileExtensions.length == 0 ? getDefaultMatcher() : getMatcherWithExtensions(fileExtensions);
    }

    public boolean accept(Path path) {
        return extMatcher.matches(path);
    }

    private static PathMatcher getMatcherWithExtensions(String... fileExStrings) {
        StringBuilder builder = new StringBuilder();
        String extGroup = Stream
                .of(fileExStrings)
                .collect(Collectors.joining("|", "\\.(?<ext>", ")"));
        builder
                .append("regex:")
                .append(".*")
                .append(extGroup);
        return FileSystems
                .getDefault()
                .getPathMatcher(builder.toString());
    }

    private static PathMatcher getDefaultMatcher() {
        return (path) -> true;
    }
}