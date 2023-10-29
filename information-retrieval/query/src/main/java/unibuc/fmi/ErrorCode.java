package unibuc.fmi;

public enum ErrorCode {
    TEST(4, "The path is not readable or writable.");

    private final String message;
    private final int code;

    private ErrorCode(int code, String message) {
        this.message = message;
        this.code = code;
    }

    public ErrorCode showError(String reason) {
        System.err.println(this.message + " " + "Reason: " + reason);
        return this;
    }

    public ErrorCode showError() {
        System.err.println(this.message);
        return this;
    }

    public void exit(String reason) {
        System.err.println(this.message + " " + "Reason: " + reason);
        System.exit(code);
    }

    public void exit() {
        System.err.println(this.message);
        System.exit(code);
    }

    public int getCode() {
        return code;
    }

    public String getMessage() {
        return message;
    }
}
