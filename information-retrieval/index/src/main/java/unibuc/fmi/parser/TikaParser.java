package unibuc.fmi.parser;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.Optional;

import org.apache.tika.mime.MediaType;
import org.apache.tika.config.TikaConfig;
import org.apache.tika.metadata.Metadata;
import org.apache.tika.parser.ParseContext;
import org.apache.tika.sax.BodyContentHandler;

import unibuc.fmi.common.Utils;
import unibuc.fmi.document.DocumentContent;

import org.apache.tika.exception.TikaException;
import org.apache.tika.parser.AutoDetectParser;

public class TikaParser {
    private static final int TIKA_NO_CHAR_LIMIT = -1;
    private final ParseContext parseContext;
    private final TikaConfig config;

    public TikaParser(TikaConfig config) {
        this.parseContext = new ParseContext();
        this.config = config;
    }

    public TikaParser() throws TikaException, IOException {
        this(new TikaConfig());
    }

    public Optional<DocumentContent> parse(Path path) {
        System.out.println(path);
        BodyContentHandler handler = new BodyContentHandler(TIKA_NO_CHAR_LIMIT);
        AutoDetectParser parser = new AutoDetectParser(config);
        Metadata metadata = new Metadata();

        if (Utils.IsDebug) {
            System.out.println("[Debug] Now parsing: " + path.toAbsolutePath().toString());
        }

        // First pass: detect file type
        MediaType mediaType;
        try (InputStream is = new BufferedInputStream(new FileInputStream(path.toFile()))) {
            mediaType = parser.getDetector().detect(is, metadata);
        } catch (Exception e) {
            if (Utils.IsDebug) {
                System.err.println("[Debug] Tika cannot detect file type: " + path.toAbsolutePath());
                System.err.println("[Debug] Error: " + e.getMessage());
            }
            return Optional.empty();
        }

        // Second pass: read input
        try (InputStream is = new BufferedInputStream(new FileInputStream(path.toFile()))) {
            parser.parse(is, handler, metadata, parseContext);
            return Optional.of(new DocumentContent(path, handler, metadata, mediaType));
        } catch (Exception e) {
            if (Utils.IsDebug) {
                System.err.println("[Debug] Tika cannot parse file: " + path.toAbsolutePath());
                System.err.println("[Debug] Error: " + e.getMessage());
            }
            return Optional.empty();
        }
    }
}
