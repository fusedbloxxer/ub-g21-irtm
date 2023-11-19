package unibuc.fmi.parse;

import java.nio.file.Path;
import java.nio.file.Paths;

import org.apache.tika.metadata.Metadata;
import org.apache.tika.mime.MediaType;
import org.xml.sax.ContentHandler;

public class TikaContent {
    private static final String METADATA_FILEPATH = "METADATA_FILEPATH";
    private static final String METADATA_FILENAME = "METADATA_FILENAME";
    private final MediaType mediaType;
    private final Metadata metadata;
    private final boolean hasError;
    private final String text;

    public TikaContent(Path path, ContentHandler handler, Metadata metadata, MediaType mediaType) {
        this.hasError = false;
        this.metadata = metadata;
        this.mediaType = mediaType;
        this.text = handler.toString();
        this.metadata.set(METADATA_FILENAME, path.getFileName().toString());
        this.metadata.set(METADATA_FILEPATH, path.toAbsolutePath().toString());
    }

    public String getText() {
        return text;
    }

    public Metadata getMetadata() {
        return metadata;
    }

    public MediaType getMediaType() {
        return mediaType;
    }

    public Path getFilepath() {
        return Paths.get(metadata.get(METADATA_FILEPATH));
    }

    public String getFilename() {
        return metadata.get(METADATA_FILENAME);
    }

    public boolean hasError() {
        return hasError;
    }
}
