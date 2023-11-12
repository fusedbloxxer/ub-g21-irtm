package unibuc.fmi.file;

import java.nio.file.Path;

@FunctionalInterface
public interface PathFilter {
    public boolean accept(Path path);
}
