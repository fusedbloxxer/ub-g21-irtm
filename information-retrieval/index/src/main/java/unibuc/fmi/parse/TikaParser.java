package unibuc.fmi.parse;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.Optional;

import org.apache.tika.config.TikaConfig;
import org.apache.tika.metadata.Metadata;
import org.apache.tika.mime.MediaType;
import org.apache.tika.parser.ParseContext;
import org.apache.tika.sax.BodyContentHandler;

import unibuc.fmi.common.Utils;

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

    public Optional<TikaContent> parse(Path path) {
        BodyContentHandler handler = new BodyContentHandler(TIKA_NO_CHAR_LIMIT);
        AutoDetectParser parser = new AutoDetectParser(config);
        Metadata metadata = new Metadata();

        try (FileInputStream fs = new FileInputStream(path.toFile()); InputStream bs = new BufferedInputStream(fs)) {
            MediaType mType = parser.getDetector().detect(bs, metadata);
            parser.parse(bs, handler, metadata, parseContext);
            return Optional.of(new TikaContent(path, handler, metadata, mType));
        } catch (Exception e) {
            if (Utils.IsDebug) {
                System.err.println("Tika cannot parse " + path.toAbsolutePath());
                System.err.println("Error: " + e.getMessage());
            }
            return Optional.empty();
        }
    }
}
