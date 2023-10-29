package unibuc.fmi;

import java.nio.file.FileSystems;
import java.nio.file.FileVisitOption;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.PathMatcher;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.Callable;
import java.util.stream.Collector;
import java.util.stream.Collectors;

import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

@Command(name = "index", version = "1.0.0", mixinStandardHelpOptions = true, exitCodeList = {})
public class IndexCommand implements Callable<Integer> {
    @Parameters(index = "0", arity = "1", paramLabel = "PATH", description = "the path to a regular or directory file")
    Path path;

    @Option(names = "--file-type", paramLabel = "FILETYPE", defaultValue = ".txt,.pdf,.docx", split = ",", description = "supported filetype formats (default: ${DEFAULT-VALUE})")
    String[] supportedFileTypes;

    @Override
    public Integer call() throws Exception {
        // Ensure that the path exists and is accessible
        if (!Files.exists(path)) {
            return ErrorCode.PATH_NOT_EXISTS
                    .showError(path.toString())
                    .getCode();
        } else if (!Files.isReadable(path) || !Files.isWritable(path)) {
            return ErrorCode.PATH_NOT_ACCESSIBLE
                    .showError(path.toString())
                    .getCode();
        }

        // Create an extension matcher for the specified fileTypes
        var fileTypes = Arrays.asList(supportedFileTypes).stream();
        var fileTypesPattern = fileTypes.map(x -> x.replaceAll("\\.", "\\\\.")).collect(Collectors.joining("|"));
        var matcher = FileSystems.getDefault().getPathMatcher("regex:" + ".*(" + fileTypesPattern + ")");

        // Consider only the supported fileTypes
        var supportedFiles = new ArrayList<Path>();
        if (Files.isDirectory(path)) {
            Files
                .find(path, Integer.MAX_VALUE, (file, attr) -> attr.isRegularFile() && matcher.matches(file))
                .collect(Collectors.toCollection(() -> supportedFiles));
        } else if (Files.isRegularFile(path)) {
            if (matcher.matches(path)) {
                supportedFiles.add(path);
            }
        } else {
            return ErrorCode.INVALID_FILE_TYPE
                .showError()
                .getCode();
        }

        // Show files
        System.out.println(supportedFiles.stream().map(x -> x.toString()).collect(Collectors.joining("\n")));
        return 0;
    }
}
