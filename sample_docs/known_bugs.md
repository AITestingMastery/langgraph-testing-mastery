# Known Bugs

## BUG-101 — Login button unresponsive on Chrome
- Severity: High
- Environment: Chrome 120+, Windows 11
- The login button occasionally does nothing on first click; a second click works.
- Suspected cause: a race condition in the form's onSubmit handler.
- Status: Open

## BUG-087 — Session not expiring on reports page
- Severity: Medium
- Environment: All browsers
- After 30 minutes idle, most pages redirect to login, but the reports page stays active.
- Status: In Progress

## BUG-112 — Password reset email delayed
- Severity: Medium
- Environment: Production only
- Reset emails sometimes take 10+ minutes instead of under 1 minute.
- Status: Open
