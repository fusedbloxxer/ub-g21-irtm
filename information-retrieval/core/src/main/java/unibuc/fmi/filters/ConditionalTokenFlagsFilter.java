package unibuc.fmi.filters;

import java.io.IOException;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.function.Function;

import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.miscellaneous.ConditionalTokenFilter;

import unibuc.fmi.attributes.TokenFlagsAttribute;
import unibuc.fmi.attributes.TokenFlagsAttribute.TokenFlag;

public class ConditionalTokenFlagsFilter extends ConditionalTokenFilter {
    private final TokenFlagsAttribute tokenFlagsAttr;
    private final EnumSet<TokenFlag> markedFlags;
    private final boolean include;

    /**
     * Applies the wrapped TokenFilters to the tokens if they have certain flags,
     * otherwise it just forwards them.
     *
     * @param input        The incoming TokenStream.
     * @param inputFactory The incoming TokenStream may be wrapped by TokenFilters
     *                     that apply when shouldFilter() matches.
     * @param include      If true and the token has at least one of the tokenFlags,
     *                     apply the TokenFilters from the inputFactory. Otherwise,
     *                     if it is false apply the inputFactory only over the
     *                     tokens which do not have one of the tokenFlags.
     * @param tokenFlags   A list of flags that may be matched by one of the tokens
     *                     using the TokenFlagsAttribute set by a
     *                     PatternTokenFlagsFilter.
     */
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

    /**
     * Applies the wrapped TokenFilters to the tokens if they have certain flags,
     * otherwise it just forwards them.
     */
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
