package unibuc.fmi.file;

@FunctionalInterface
public interface Filter<T> {
    public boolean accept(T item);
}
