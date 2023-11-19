package unibuc.fmi.analyze.attributes;

import java.util.EnumSet;

import org.apache.lucene.util.Attribute;

public interface TokenFlagsAttribute extends Attribute {
    public static final TokenFlag DEFAULT_TOKEN_FLAG = TokenFlag.Word;

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

    public void setTokenFlags(EnumSet<TokenFlag> tokenFlags);

    public EnumSet<TokenFlag> getTokenFlags();

    public boolean isFinalToken();
}
