package unibuc.fmi.analyzer;

public class AnalyzerOptions {
    private boolean useSynonyms;
    private AnalyzerMode mode;

    /**
     * @param mode        Inform the analyzer if it's being used for querying or
     *                    indexing;
     *                    One difference is that the index mode can also store
     *                    synonyms;
     * @param useSynonyms Inject single- or multi-tokens at index time;
     *                    May produce false positives and increase indexing time;
     */
    public AnalyzerOptions(AnalyzerMode mode, boolean useSynonyms) {
        this.useSynonyms = useSynonyms;
        this.mode = mode;
    }

    public boolean usesSynonyms() {
        return useSynonyms;
    }

    public void useSynonyms(boolean useSynonyms) {
        this.useSynonyms = useSynonyms;
    }

    public boolean isIndexing() {
        return mode == AnalyzerMode.INDEXING;
    }

    public boolean isQuerying() {
        return mode == AnalyzerMode.QUERYING;
    }

    public AnalyzerMode getMode() {
        return mode;
    }

    public void setMode(AnalyzerMode mode) {
        this.mode = mode;
    }
}
