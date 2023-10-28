package unibuc.fmi;

import org.apache.commons.collections.MapIterator;
import org.apache.commons.collections.map.HashedMap;

public class App {
    public static void main(String[] args) {
        System.out.println("Index.");
        HashedMap map = new HashedMap();
        map.put("andrei", "are");
        MapIterator it = map.mapIterator();
        while (it.hasNext()) {
            Object key = it.next();
            Object value = it.getValue();
            System.out.println(key + " " + value);
        }
    }
}