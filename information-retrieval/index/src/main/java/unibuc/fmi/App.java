package unibuc.fmi;

import picocli.CommandLine;

public class App {
    public static void main(String... args) {
        System.exit(new CommandLine(new IndexCommand()).execute(args));
    }
}
