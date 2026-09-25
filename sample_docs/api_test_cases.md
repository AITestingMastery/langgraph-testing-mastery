# API Test Cases

## POST /api/login
- Happy path: valid creds -> 200 with JWT + refresh token.
- Missing fields -> 400 with validation message.
- Wrong creds -> 401, no token.
- Rate limiting: 20 requests / 10s -> 429 with Retry-After.

## GET /api/orders
- Authorized (valid token) -> 200, only the user's orders.
- Unauthorized (no token) -> 401.
- Cross-user access -> 403, must not leak another user's data.

## Performance
- Orders endpoint should respond under 300ms p95.
- Login p95 has spiked above 800ms during peak load.
