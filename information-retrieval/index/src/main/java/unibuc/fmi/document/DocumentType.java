package unibuc.fmi.document;

public enum DocumentType {
    DOCX("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"),
    DOC("application/x-tika-msoffice", "doc"),
    PDF("application/pdf", "pdf"),
    TXT("text/plain", "txt");

    private final String mediaType;
    private final String suffix;

    private DocumentType(String mediaType, String suffix) {
        this.mediaType = mediaType;
        this.suffix = suffix;
    }

    public String getMediaType() {
        return this.mediaType;
    }

    public String getSuffix() {
        return this.suffix;
    }

    @Override
    public String toString() {
        return this.mediaType;
    }
}
