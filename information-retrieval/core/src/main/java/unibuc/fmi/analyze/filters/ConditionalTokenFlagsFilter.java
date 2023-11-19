package unibuc.fmi.analyze.filters;

import java.io.IOException;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.function.Function;

import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.miscellaneous.ConditionalTokenFilter;

import unibuc.fmi.analyze.attributes.TokenFlagsAttribute;
import unibuc.fmi.analyze.attributes.TokenFlagsAttribute.TokenFlag;

public class ConditionalTokenFlagsFilter extends ConditionalTokenFilter {
    private final TokenFlagsAttribute tokenFlagsAttr;
    private final EnumSet<TokenFlag> markedFlags;
    private final boolean include;

    public ConditionalTokenFlagsFilter(TokenStream input, Function<TokenStream, TokenStream> inputFactory,
            boolean include, TokenFlag... tokenFlags) {
        this(input, inputFactory, include, tokenFlags.length == 0
                ? EnumSet.noneOf(TokenFlag.class)
                : EnumSet.copyOf(Arrays.asList(tokenFlags)));
    }

    protected ConditionalTokenFlagsFilter(TokenStream input, Function<TokenStream, TokenStream> inputFactory,
            boolean include, EnumSet<TokenFlag> tokenFlags) {
        super(input, inputFactory);
        tokenFlagsAttr = input.addAttribute(TokenFlagsAttribute.class);
        this.markedFlags = tokenFlags;
        this.include = include;
    }

    @Override
    protected boolean shouldFilter() throws IOException {
        if (markedFlags.isEmpty()) {
            return this.include;
        }

        if (tokenFlagsAttr.getTokenFlags().stream().anyMatch(x -> markedFlags.contains(x))) {
            return this.include;
        }

        return !this.include;
    }
}
