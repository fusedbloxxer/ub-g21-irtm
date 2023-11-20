package unibuc.fmi.filters;

import java.util.Arrays;
import java.util.List;

import org.apache.tika.mime.MediaType;

import unibuc.fmi.document.DocumentType;

public class DocumentTypeFilter implements Filter<MediaType> {
    private final List<DocumentType> allowedDocTypes;

    /**
     * Create a filter that checks the MediaType to match one of the DocumentTypes.
     *
     * @param types
     */
    public DocumentTypeFilter(DocumentType... types) {
        this.allowedDocTypes = Arrays.asList(types);
    }

    public DocumentTypeFilter() {
        this(DocumentType.DOC, DocumentType.DOCX, DocumentType.PDF, DocumentType.TXT);
    }

    /**
     * Filter out documents which have a different MediaType than one of the
     * specified DocumentTypes.
     */
    @Override
    public boolean accept(MediaType item) {
        return allowedDocTypes
                .stream()
                .anyMatch(x -> x.toString().equals(item.toString()));
    }
}
