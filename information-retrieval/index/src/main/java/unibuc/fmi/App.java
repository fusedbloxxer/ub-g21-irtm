package unibuc.fmi;

import picocli.CommandLine;
import unibuc.fmi.command.IndexCommand;

public class App {
    public static void main(String... args) {
        System.exit(new CommandLine(new IndexCommand()).execute(args));
    }
}
