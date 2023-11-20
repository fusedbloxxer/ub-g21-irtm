package unibuc.fmi.filters;

@FunctionalInterface
public interface Filter<T> {
    public boolean accept(T item);
}
