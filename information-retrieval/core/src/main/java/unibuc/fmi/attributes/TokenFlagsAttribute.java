package unibuc.fmi.attributes;

import java.util.EnumSet;

import org.apache.lucene.util.Attribute;

public interface TokenFlagsAttribute extends Attribute {
    public static final TokenFlag DEFAULT_TOKEN_FLAG = TokenFlag.Word;

    /**
     * Flags which indicate the content of a token's text.
     */
    public enum TokenFlag {
        EmailAddress,
        NamedEntity,
        Punctuation,
        PhoneNumber,
        UrlAddress,
        Acronym,
        Number,
        Roman,
        Word,
        Date,
    }

    /**
     * Should be final when a token has a single non-default flag.
     */
    public boolean isFinalToken();

    public EnumSet<TokenFlag> getTokenFlags();

    public void setTokenFlags(EnumSet<TokenFlag> tokenFlags);
}
