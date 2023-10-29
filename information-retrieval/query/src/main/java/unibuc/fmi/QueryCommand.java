package unibuc.fmi;

import java.util.concurrent.Callable;

import picocli.CommandLine.Command;

@Command(name = "query", version = "1.0.0", mixinStandardHelpOptions = true)
public class QueryCommand implements Callable<Integer> {
    @Override
    public Integer call() {
        System.out.println("hello");
        return 0;
    }
}
