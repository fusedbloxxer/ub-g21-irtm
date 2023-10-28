package unibuc.fmi;

import org.apache.commons.collections4.map.HashedMap;

public class App {
    public static void main(String[] args) {
        System.out.println("Index.");
        HashedMap<String, String> m = new HashedMap<>();
        m.clear();
        Useful.Hey();
    }
}