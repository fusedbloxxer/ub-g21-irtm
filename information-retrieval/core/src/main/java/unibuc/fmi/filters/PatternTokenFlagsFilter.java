package unibuc.fmi.filters;

import java.io.IOException;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;

import unibuc.fmi.attributes.TokenFlagsAttribute;
import unibuc.fmi.attributes.TokenFlagsAttribute.TokenFlag;
import unibuc.fmi.attributes.impl.TokenFlagsAttributeImpl;

public class PatternTokenFlagsFilter extends TokenFilter {
    private final List<TokenFlagsPatternRule> rules;
    private final TokenFlagsAttribute tokenClassAttr;
    private final CharTermAttribute termAttr;

    /**
     * Determine if the content of a token is one of the possible TokenFlag values
     * by applying RegexPatterns. This, in turns sets the TokenFlagsAttribute for
     * each token.
     *
     * @param input A stream of tokens.
     */
    public PatternTokenFlagsFilter(TokenStream input) {
        super(input);

        // Add custom attribute
        input.addAttributeImpl(new TokenFlagsAttributeImpl());

        // Add existing attributes
        tokenClassAttr = input.addAttribute(TokenFlagsAttribute.class);
        termAttr = input.addAttribute(CharTermAttribute.class);

        // Source:
        // https://stackoverflow.com/questions/201323/how-can-i-validate-an-email-address-using-a-regular-expression
        var emailRule = new TokenFlagsPatternRule("[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+",
                TokenFlagsAttribute.TokenFlag.EmailAddress);
        // Source:
        // https://stackoverflow.com/questions/15491894/regex-to-validate-date-formats-dd-mm-yyyy-dd-mm-yyyy-dd-mm-yyyy-dd-mmm-yyyy
        var dateRule = new TokenFlagsPatternRule(
                "(?:(?:31(\\/|-|\\.)(?:0?[13578]|1[02]))\\1|(?:(?:29|30)(\\/|-|\\.)(?:0?[13-9]|1[0-2])\\2))(?:(?:1[6-9]|[2-9]\\d)?\\d{2})$|^(?:29(\\/|-|\\.)0?2\\3(?:(?:(?:1[6-9]|[2-9]\\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00))))$|^(?:0?[1-9]|1\\d|2[0-8])(\\/|-|\\.)(?:(?:0?[1-9])|(?:1[0-2]))\\4(?:(?:1[6-9]|[2-9]\\d)?\\d{2})",
                TokenFlagsAttribute.TokenFlag.Date);
        // Source:
        // https://ihateregex.io/expr/phone/
        var phoneRule = new TokenFlagsPatternRule("[\\+]?[(]?[0-9]{3}[)]?[-\\s\\.]?[0-9]{3}[-\\s\\.]?[0-9]{4,6}",
                TokenFlagsAttribute.TokenFlag.PhoneNumber);
        // Source:
        // https://stackoverflow.com/questions/5457416/how-to-validate-numeric-values-which-may-contain-dots-or-commas
        var numberRule = new TokenFlagsPatternRule("[0-9]{1,2}([,.][0-9]{1,2})?",
                TokenFlagsAttribute.TokenFlag.Number);
        // Source:
        // https://stackoverflow.com/questions/29586972/match-only-roman-numerals-with-regular-expression
        var romanRule = new TokenFlagsPatternRule("(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})",
                TokenFlagsAttribute.TokenFlag.Roman);
        // Source:
        // https://www.freecodecamp.org/news/how-to-write-a-regular-expression-for-a-url/
        var urlRule = new TokenFlagsPatternRule(
                "(https:\\/\\/www\\.|http:\\/\\/www\\.|https:\\/\\/|http:\\/\\/)?[a-zA-Z]{2,}(\\.[a-zA-Z]{2,})(\\.[a-zA-Z]{2,})?\\/[a-zA-Z0-9]{2,}|((https:\\/\\/www\\.|http:\\/\\/www\\.|https:\\/\\/|http:\\/\\/)?[a-zA-Z]{2,}(\\.[a-zA-Z]{2,})(\\.[a-zA-Z]{2,})?)|(https:\\/\\/www\\.|http:\\/\\/www\\.|https:\\/\\/|http:\\/\\/)?[a-zA-Z0-9]{2,}\\.[a-zA-Z0-9]{2,}\\.[a-zA-Z0-9]{2,}(\\.[a-zA-Z0-9]{2,})?",
                TokenFlagsAttribute.TokenFlag.UrlAddress);
        // Source:
        // https://stackoverflow.com/questions/35076016/regex-to-match-acronyms
        var acrynomRule = new TokenFlagsPatternRule("(?:[a-zA-Z]\\.){2,}",
                TokenFlagsAttribute.TokenFlag.Acronym);

        // Accumulate all rules
        this.rules = Arrays.asList(emailRule, dateRule, phoneRule, numberRule, romanRule, urlRule, acrynomRule);
    }

    @Override
    public boolean incrementToken() throws IOException {
        if (!input.incrementToken()) {
            return false;
        }

        // Apply each pattern and aggregate their flags
        EnumSet<TokenFlag> tokenFlags = rules
                .stream()
                .filter(x -> x.getPattern().matcher(this.termAttr).matches())
                .flatMap(x -> x.getTokenFlags().stream())
                .collect(Collectors.toCollection(() -> EnumSet.noneOf(TokenFlag.class)));

        // Save the detected types for the token
        if (!tokenFlags.isEmpty()) {
            tokenClassAttr.setTokenFlags(tokenFlags);
        }

        // Continue with the rest of the tokenStream
        return true;
    }

    public static class TokenFlagsPatternRule {
        private final EnumSet<TokenFlag> tokenFlags;
        private final Pattern pattern;

        /**
         * Helper class used to hold a regex that when matched againts token text should
         * indicate that the token should have a list of flags.
         *
         * @param regex      A pattern that represents the given flags.
         * @param tokenFlags Indicate what a token should represent when matched.
         */
        public TokenFlagsPatternRule(String regex, TokenFlag... tokenFlags) {
            this.pattern = Pattern.compile(regex);
            this.tokenFlags = tokenFlags.length == 0
                    ? EnumSet.noneOf(TokenFlag.class)
                    : EnumSet.copyOf(Arrays.asList(tokenFlags));
        }

        public EnumSet<TokenFlag> getTokenFlags() {
            return this.tokenFlags;
        }

        public Pattern getPattern() {
            return this.pattern;
        }
    }
}
