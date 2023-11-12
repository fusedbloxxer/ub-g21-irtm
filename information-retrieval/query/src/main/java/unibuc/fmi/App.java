package unibuc.fmi;

import picocli.CommandLine;
import unibuc.fmi.command.QueryCommand;

public class App {
    public static void main(String... args) {
        System.exit(new CommandLine(new QueryCommand()).execute(args));
    }
}