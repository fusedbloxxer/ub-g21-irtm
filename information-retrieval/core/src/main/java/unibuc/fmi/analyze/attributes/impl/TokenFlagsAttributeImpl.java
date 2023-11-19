package unibuc.fmi.analyze.attributes.impl;

import java.util.EnumSet;

import org.apache.lucene.util.AttributeImpl;
import org.apache.lucene.util.AttributeReflector;

import unibuc.fmi.analyze.attributes.TokenFlagsAttribute;

public class TokenFlagsAttributeImpl extends AttributeImpl implements TokenFlagsAttribute {
    private EnumSet<TokenFlag> tokenFlags;

    @Override
    public void setTokenFlags(EnumSet<TokenFlag> tokenFlags) {
        this.tokenFlags = tokenFlags;
    }

    @Override
    public EnumSet<TokenFlag> getTokenFlags() {
        return tokenFlags;
    }

    @Override
    public boolean isFinalToken() {
        if (tokenFlags.size() == 1) {
            if (!tokenFlags.contains(DEFAULT_TOKEN_FLAG)) {
                return true;
            } else {
                return false;
            }
        } else {
            return false;
        }
    }

    @Override
    public void clear() {
        this.tokenFlags = EnumSet.of(TokenFlagsAttribute.DEFAULT_TOKEN_FLAG);
    }

    @Override
    public void reflectWith(AttributeReflector reflector) {
        reflector.reflect(TokenFlagsAttribute.class, "tokenFlags", getTokenFlags());
    }

    @Override
    public void copyTo(AttributeImpl target) {
        ((TokenFlagsAttribute) target).setTokenFlags(tokenFlags);
    }
}
