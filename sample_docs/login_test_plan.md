# Login Test Plan

## Scope
Authentication for the web app: sign-in, sign-out, session handling, password reset.

## Test Cases
- Valid login: registered email + correct password -> dashboard within 2s.
- Invalid password: inline error, no redirect.
- Locked account: 5 wrong attempts -> locks 15 minutes.
- Session timeout: 30 minutes idle -> redirect to login with "session expired".
- Password reset: reset email within 1 minute, link valid 60 minutes.

## Known Risks
- Login button intermittently unresponsive on Chrome (see known bugs).
