package unibuc.fmi.file;

import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import unibuc.fmi.filters.Filter;
import unibuc.fmi.filters.PathSuffixFilter;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.AccessDeniedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

public class PathSearch {
    public static class PathSearchOptions {
        String[] fileExtensions;

        public PathSearchOptions(String... fileExtensions) {
            this.fileExtensions = fileExtensions;
        }
    }

    public static List<Path> from(Path path, PathSearchOptions options) throws IOException {
        if (!Files.exists(path)) {
            throw new FileNotFoundException(path.toAbsolutePath().toString());
        }

        if (!Files.isReadable(path)) {
            throw new AccessDeniedException(path.toAbsolutePath().toString());
        }

        Filter<Path> pathFilter = new PathSuffixFilter(options.fileExtensions);
        Predicate<Path> isValidFile = x -> Files.isReadable(x) && Files.isRegularFile(x);
        Predicate<Path> isExtAllowed = x -> pathFilter.accept(x);
        Predicate<Path> isAllowed = isValidFile.and(isExtAllowed);

        if (Files.isRegularFile(path)) {
            return Arrays
                    .asList(path)
                    .stream()
                    .filter(isAllowed)
                    .collect(Collectors.toList());
        }

        try (Stream<Path> pathWalk = Files.walk(path)) {
            return pathWalk
                    .filter(isAllowed)
                    .collect(Collectors.toList());
        }
    }
}
