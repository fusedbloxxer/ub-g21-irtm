package unibuc.fmi.common;

public enum DocumentTypes {
    DOCX("application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    PDF("application/pdf"),
    TXT("text/plain");

    DocumentTypes(String mediaType) {
        this.mediaType = mediaType;
    }

    private final String mediaType;

    @Override
    public String toString() {
        return this.mediaType;
    }
}
