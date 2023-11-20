package unibuc.fmi.filters;

import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.nio.file.PathMatcher;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class PathSuffixFilter implements Filter<Path> {
    private final PathMatcher extMatcher;

    /**
     * Create a filter that matches a Path's extension to a list of allowed file
     * extensions. If no extensions is given, then match any Pparath.
     *
     * @param fileExtensions File extensions: pdf, txt, etc.
     */
    public PathSuffixFilter(String... fileExtensions) {
        extMatcher = fileExtensions.length == 0 ? getDefaultMatcher() : getMatcherWithExtensions(fileExtensions);
    }

    public boolean accept(Path path) {
        return extMatcher.matches(path);
    }

    /**
     * Create a regex that matches any of the file extensions.
     *
     * @param fileExStrings File extendsions: pdf, txt, etc.
     */
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

    /**
     * Dummy matcher that accepts any path.
     */
    private static PathMatcher getDefaultMatcher() {
        return (path) -> true;
    }
}