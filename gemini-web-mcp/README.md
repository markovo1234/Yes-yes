# gemini-web-mcp

A personal remote MCP server that generates and edits images with **your Gemini web account**
(Google AI Pro, logged in through browser cookies). You add it to claude.ai as a custom connector,
and it works on web and phone. It can make several images per call, and when you say "end chat" it
deletes those chats from your Gemini history.

- **Tools:** `generate_images`, `edit_image`, `end_chat`. There are no others.
- **Routes:** `POST|GET|DELETE /mcp/<MCP_PATH_SECRET>` (MCP, Streamable HTTP, stateless) and
  `GET /i/<session_id>/<image_id>.<ext>` (image downloads). Every other path returns **404**, and so does a
  wrong or missing secret. The secret is checked in constant time before anything else runs.
- **Storage:** everything lives on the Railway volume at `/data`, owner-only (mode 0600/0700):
  `cookies.json`, `state.db` (SQLite: sessions, their Gemini chat ids, images, hourly counter) and
  `images/<session_id>/` (full images + 384 px previews).
- **Gemini access:** [`gemini_webapi`](https://github.com/HanaokaYuzu/Gemini-API) pinned to **2.1.1**,
  MCP Python SDK pinned to **2.2.0**. All dependencies are pinned in `requirements.txt`.

## How it behaves

| | |
|---|---|
| Concurrency | At most 2 Gemini requests at a time (generation, edits and deletes share the limit). |
| Attempts | At most 2 per request. The library's own retries (up to 6 sends) are switched off. |
| Deadline | 200 s per tool call. Finished images are returned and the rest are marked "timed out". |
| Hourly limit | `MAX_IMAGES_PER_HOUR` (default 30), over a rolling hour, stored in `state.db` so it survives restarts. |
| Auth/account problems | One clear "refresh your cookies" message. Gemini is not contacted again until the service restarts. |
| Usage limit / IP block | Generation stops right away with no retries, and pauses for 30 minutes. |
| Refusals | Reported word for word. The server never rewrites a prompt. |
| Result size | Text first, then one JPEG preview per image (long side ≤ 384 px). The whole result is kept under 120,000 characters: previews shrink first, then get dropped. |
| Cleanup | Every 30 minutes, sessions idle longer than `SESSION_TTL_HOURS` (default 24) get the full `end_chat` treatment. |
| Logs | Only ids, timings, counts and error types. Never prompts, Gemini text, image data, cookies or the path secret. uvicorn's access log is off, and a redacted line is logged instead. `gemini_webapi`'s own logger is muted. |

Sessions: calling `generate_images` without `session_id` creates a random 128-bit id. Every Gemini
chat created for that session is recorded as soon as Gemini assigns its id, including chats from failed
attempts. `end_chat` can therefore remove all of them from your Gemini history.

`edit_image` continues the source image's own Gemini chat, from that image's turn. If the source turn
produced several images, or the chat has moved on since, the exact picture is also attached so Gemini
edits the right one. The original image and its link stay.

## Environment variables

| Variable | Required | Notes |
|---|---|---|
| `MCP_PATH_SECRET` | yes | base64url, at least 32 random bytes (43+ chars). The server refuses to start otherwise. |
| `GEMINI_1PSID` | yes (first boot) | Seed for `__Secure-1PSID`. |
| `GEMINI_1PSIDTS` | yes (first boot) | Seed for `__Secure-1PSIDTS`. |
| `PORT` | Railway | Set to `8080` below. |
| `DATA_DIR` | no | Default `/data`. |
| `PUBLIC_BASE_URL` | no | Base for image links. On Railway it defaults to `https://$RAILWAY_PUBLIC_DOMAIN`. |
| `MAX_IMAGES_PER_HOUR` | no | Default 30. |
| `SESSION_TTL_HOURS` | no | Default 24. |
| `GEMINI_MODEL` | no | For example `gemini-pro`. Empty means your account's default model. An unknown name gives a clear error listing the models your account offers. |

### How cookies are handled

The two variables are only the **seed**. On boot, if `/data/cookies.json` exists it is used.
Otherwise the seed is used. After a successful login, and every time the library refreshes
`__Secure-1PSIDTS` (about every 10 minutes), the cookies are written to `/data/cookies.json` with mode 0600.
Refreshed cookies therefore survive restarts and redeploys.
`cookies.json` also records a fingerprint of the seed it grew from. **If you change the seed variables,
the next boot uses the new seed automatically**, even if an old `cookies.json` is still on the volume.

## Getting the two cookies from gemini.google.com

Firefox is recommended. The library notes that Chromium's "Device Bound Session Credentials" can make
exported cookies expire within hours and prevent them from being renewed.

1. Open a **private window**, go to <https://gemini.google.com> and sign in with your AI Pro account.
2. Press **F12**, open **Storage** (Firefox) or **Application** (Chrome), then **Cookies** → `https://gemini.google.com`.
3. Copy the values of `__Secure-1PSID` and `__Secure-1PSIDTS`.
4. Close the private window **without signing out**. Signing out invalidates the cookies.

Treat these values like your Google password. Put them only into Railway variables (below) or your
local git-ignored `.env`. Never paste them into a chat.

## Deploying to Railway

You need the Railway CLI. Either run `npm i -g @railway/cli` or `bash <(curl -fsSL railway.com/install.sh)`.

```bash
cd gemini-web-mcp
railway login                      # or: railway login --browserless
railway init                       # create a project, e.g. "gemini-web-mcp"
railway add --service gemini-web-mcp
railway link                       # pick the project + the gemini-web-mcp service
railway volume add --mount-path /data

# Variables. --stdin keeps the values out of your shell history: paste the value, press Enter, then Ctrl-D.
python3 -c "import secrets,base64;print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip('='))" \
  | railway variable set MCP_PATH_SECRET --stdin --skip-deploys
railway variable set GEMINI_1PSID --stdin --skip-deploys
railway variable set GEMINI_1PSIDTS --stdin --skip-deploys
railway variable set PORT=8080 MAX_IMAGES_PER_HOUR=30 SESSION_TTL_HOURS=24 --skip-deploys

railway domain --port 8080         # creates https://<name>.up.railway.app
railway up --detach                # builds the Dockerfile and deploys
railway logs                       # expect: "gemini client ready" and "cookies saved to cookies.json"
```

`railway.json` sets 1 replica, **serverless/app sleeping off** (`sleepApplication: false`), and restart on
failure. After the first deploy, open the service's **Settings** in the Railway dashboard and confirm that
*Serverless* is off and replicas = 1.

The connector URL is:

```
https://<name>.up.railway.app/mcp/<MCP_PATH_SECRET>
```

To see the secret again, run `railway variable list` or open the dashboard's Variables tab. The URL is a
password: anyone who has it can use your Gemini account through this server.

## Adding the connector in claude.ai

1. In claude.ai go to **Customize → Connectors → + → Add custom connector**.
2. Name: `Gemini images`. URL: the connector URL above. Leave OAuth fields empty. Then choose **Add**.
3. In a chat, open the **+** / tools menu → **Connectors** and switch **Gemini images** on.
   Connectors you add on the web also appear in the Claude mobile apps.
4. Try "make me 3 images of a lighthouse in a storm". Say "end chat" when you're done to delete the
   chats from Gemini and kill the links.

## Refreshing cookies

When Claude reports *"Gemini says the saved cookies are not signed in … Refresh your cookies"*:

1. Get fresh values as in [Getting the two cookies](#getting-the-two-cookies-from-geminigooglecom).
2. Update the variables. Setting a variable triggers a redeploy:
   ```bash
   railway variable set GEMINI_1PSID --stdin --skip-deploys
   railway variable set GEMINI_1PSIDTS --stdin
   ```
3. Because the seed changed, the new boot ignores the old `/data/cookies.json` and overwrites it.
   If you want to remove the old file yourself (for example when you re-enter the *same* values), delete it
   and restart:
   ```bash
   railway ssh -- rm -f /data/cookies.json
   railway redeploy
   ```
4. Check `railway logs` for `gemini client ready`.

## Rotating MCP_PATH_SECRET

```bash
python3 -c "import secrets,base64;print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip('='))" \
  | railway variable set MCP_PATH_SECRET --stdin     # triggers a redeploy
```

Then in claude.ai remove the old connector and add a new one with the new URL (**Customize → Connectors**).
The old URL returns 404 as soon as the new deployment is live.

## Running locally

```bash
cd gemini-web-mcp
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env        # fill in MCP_PATH_SECRET, GEMINI_1PSID, GEMINI_1PSIDTS (never commit .env)
set -a && . ./.env && set +a
python -m app               # http://localhost:8080/mcp/<secret>
```

Tests use a fake Gemini backend and never contact Google:

```bash
python -m pytest
# MCP Inspector against the real app with the fake backend:
MCP_PATH_SECRET=<test secret> DATA_DIR=/tmp/fake-data PORT=8765 python -m tests.run_fake_server &
npx -y @modelcontextprotocol/inspector@2.8.0 --cli http://127.0.0.1:8765/mcp/<test secret> --transport http --method tools/list
```

## Notes and caveats

- Railway's own HTTP request logs, which project members can see in the dashboard, record request paths.
  That includes `/mcp/<secret>`. This is inherent to a secret in the URL path. The app itself never logs it.
- The library sends a lightweight heartbeat every 1–2 minutes and rotates `__Secure-1PSIDTS` every
  ~10 minutes while connected, as a browser tab would. After an error it reconnects on the next request, or
  within 30 minutes via the maintenance task, so cookie refresh keeps running.
- `gemini_webapi` is a reverse-engineered client (AGPL-3.0) for Gemini's web app. Google can change the web
  app at any time. `tests/test_library_contract.py` fails loudly if the pinned library's behaviour this
  server relies on moves.
