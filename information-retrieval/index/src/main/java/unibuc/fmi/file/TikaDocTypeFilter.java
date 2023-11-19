package unibuc.fmi.file;

import java.util.Arrays;
import java.util.List;

import org.apache.tika.mime.MediaType;

import unibuc.fmi.common.DocumentType;

public class TikaDocTypeFilter implements Filter<MediaType> {
    private final List<DocumentType> allowedDocTypes;

    public TikaDocTypeFilter(DocumentType... types) {
        this.allowedDocTypes = Arrays.asList(types);
    }

    public TikaDocTypeFilter() {
        this(DocumentType.DOC, DocumentType.DOCX, DocumentType.PDF, DocumentType.TXT);
    }

    @Override
    public boolean accept(MediaType item) {
        return allowedDocTypes
            .stream()
            .anyMatch(x -> x.toString().equals(item.toString()));
    }
}
