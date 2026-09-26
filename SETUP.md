# 🧭 Step-by-Step Setup & User Guide

This guide takes you from **nothing installed** to **every feature working** — one step
at a time. Follow it top to bottom. Every command is given for **macOS** and
**Windows**.

> **Windows users:** use **Command Prompt (cmd)**, not PowerShell, unless a step says
> otherwise. Search "cmd" in the Start menu.

---

## How long will this take?

| Part | What you get | Time | Needed? |
|---|---|---|---|
| 1–5 | The app running with research, bug reports, loops, guardrails | ~15 min | ✅ **Required** |
| 6 | Real Jira tickets | ~10 min | Optional |
| 7 | Real emails via Gmail | ~15 min | Optional |
| 8 | LangSmith trace links | ~5 min | Optional |
| 9 | A guided tour of every feature | ~15 min | Recommended |

You can stop after Part 5 and add Jira, Gmail and LangSmith later. Anything not
connected is shown as 🔴 and is **skipped, never faked**.

---

## Contents

- [Part 1 — Install the tools](#part-1--install-the-tools)
- [Part 2 — Get the code](#part-2--get-the-code)
- [Part 3 — Create a virtual environment and install](#part-3--create-a-virtual-environment-and-install)
- [Part 4 — Get an OpenAI key and create `.env`](#part-4--get-an-openai-key-and-create-env)
- [Part 5 — Run the tests, then the app](#part-5--run-the-tests-then-the-app)
- [Part 6 — Connect Jira (optional)](#part-6--connect-jira-optional)
- [Part 7 — Connect Gmail (optional)](#part-7--connect-gmail-optional)
- [Part 8 — Connect LangSmith (optional)](#part-8--connect-langsmith-optional)
- [Part 9 — Guided tour: try every feature](#part-9--guided-tour-try-every-feature)
- [Part 10 — Understanding the screen](#part-10--understanding-the-screen)
- [Part 11 — Everyday commands](#part-11--everyday-commands)
- [Part 12 — Make it yours](#part-12--make-it-yours)
- [Part 13 — Troubleshooting](#part-13--troubleshooting)
- [Final checklist](#final-checklist)

---

## Part 1 — Install the tools

You need three things: **Git**, **Python 3.12**, and **uv**.

> **Why Python 3.12 exactly?** It's the version this project is tested with. Newer
> versions can break some AI libraries. If you already have 3.12, skip that step.

### 1.1 Check what you already have

**macOS** (open **Terminal**):
```bash
git --version
python3.12 --version
uv --version
```

**Windows** (cmd):
```bat
git --version
py -3.12 --version
uv --version
```

Each line should print a version number. Install only what's missing.

### 1.2 Install Git

- **macOS:** run `xcode-select --install` and click **Install** in the pop-up.
- **Windows:** download from **https://git-scm.com/download/win** and run the
  installer with the default options.

### 1.3 Install Python 3.12

- **macOS:** with [Homebrew](https://brew.sh): `brew install python@3.12`
  — or download the macOS installer for **3.12.x** from
  **https://www.python.org/downloads/macos/**.
- **Windows:** download the **3.12.x** Windows installer from
  **https://www.python.org/downloads/windows/**.
  ⚠️ On the first screen, **tick "Add python.exe to PATH"**, then click *Install Now*.

### 1.4 Install uv

`uv` is a fast Python package tool. **The Jira and Gmail servers are launched with
`uvx`** (part of uv), so you need it even if you install packages with `pip`.

- **macOS:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Windows** (this one line works from cmd):
  ```bat
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

**Close and reopen your terminal**, then run `uv --version` to confirm.

✅ **Checkpoint:** all three commands from step 1.1 print a version.

---

## Part 2 — Get the code

Pick a folder for your projects (for example your Desktop), then clone:

**macOS:**
```bash
cd ~/Desktop
git clone https://github.com/AITestingMastery/langgraph-testing-mastery.git
cd langgraph-testing-mastery
```

**Windows:**
```bat
cd %USERPROFILE%\Desktop
git clone https://github.com/AITestingMastery/langgraph-testing-mastery.git
cd langgraph-testing-mastery
```

✅ **Checkpoint:** `ls` (Mac) or `dir` (Windows) shows `app.py`, `README.md`, `SETUP.md`.

> **Every command from now on runs inside this `langgraph-testing-mastery` folder.**

---

## Part 3 — Create a virtual environment and install

A virtual environment (`.venv`) keeps this project's packages separate from the rest of
your computer.

### 3.1 Create and activate it

**macOS:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
```

Your prompt now starts with **`(.venv)`**. That means it's active.

> 🔁 **Every time you open a new terminal**, go to the project folder and run the
> *activate* line again. If `(.venv)` is missing, commands will fail with
> "module not found".

### 3.2 Install the packages

```bash
pip install -r requirements.txt
```
(Faster alternative: `uv pip install -r requirements.txt`)

This takes 1–3 minutes.

✅ **Checkpoint:** it ends without red `ERROR` lines.

---

## Part 4 — Get an OpenAI key and create `.env`

### 4.1 Get an OpenAI API key

1. Go to **https://platform.openai.com/api-keys** and sign in.
2. Click **Create new secret key**, give it a name, click **Create**.
3. **Copy the key now** (it starts with `sk-`) — you can't see it again later.
4. Make sure your account has credit: **Settings → Billing**. The default model,
   `gpt-4o-mini`, is inexpensive; a single request typically costs well under a cent,
   but check OpenAI's pricing page for current rates.

> 🔐 Treat this key like a password. Never paste it into chats, screenshots, or code
> you commit.

### 4.2 Create your `.env` file

`.env` holds your keys and settings. It is **git-ignored**, so it never gets uploaded.

**macOS:**
```bash
cp .env.example .env
open -e .env
```

**Windows:**
```bat
copy .env.example .env
notepad .env
```

### 4.3 Fill in the required value

In `.env`, replace the placeholder:
```dotenv
OPENAI_API_KEY=sk-your-key-here
```
with your real key. No quotes, no spaces around `=`.

While you're here, set **who receives "email me" requests** (you'll need this in Part 7):
```dotenv
DEFAULT_EMAIL_TO=your.name@gmail.com
```

Leave everything else as it is for now. Save and close the file.

> 📄 **Mac tip:** files starting with `.` are hidden in Finder. Press
> **Cmd + Shift + .** to show them.

✅ **Checkpoint:** `.env` exists and contains your real `OPENAI_API_KEY`.

---

## Part 5 — Run the tests, then the app

### 5.1 Run the offline tests

These use fake AI models, so they need **no keys and no internet** — they check that
your install is correct.

```bash
python -m pytest tests -q
```

✅ **Expected:** `87 passed`.

> Seeing `ModuleNotFoundError`? Your `.venv` isn't active (step 3.1), or you're not in
> the project folder.

### 5.2 Start the app

```bash
streamlit run app.py
```

Your browser opens **http://localhost:8501** (open it manually if it doesn't).

The first start takes a little longer — it launches the Jira and Gmail servers and
builds the document index. You may see lots of text in the terminal, including a big
**FastMCP** banner and some warnings. That's normal.

### 5.3 First question

1. In the main area, open **💡 Try these — click to run**.
2. Click **📚 Research only**.
3. Watch the status box show each step as it runs.
4. The answer lists the known login bugs (BUG-101 on Chrome, and related issues).

✅ **Checkpoint:** you got an answer, and under it a row like
`🔧 1 tool call · 📚 1 source · ⏱ 6.2s`.

🎉 **The core app works.** Parts 6–8 connect the optional services. You can skip
ahead to [Part 9](#part-9--guided-tour-try-every-feature) now and come back later.

> **To stop the app:** click in the terminal and press **Ctrl + C**.

---

## Part 6 — Connect Jira (optional)

This lets the app **create real Jira tickets** — always after you approve.

### 6.1 Get a Jira site

If you don't have one: go to **https://www.atlassian.com/software/jira/free**, sign up
(free plan), and choose a site name. Your site URL will look like
`https://yourname.atlassian.net`.

### 6.2 Create a project with the key `TEST`

1. In Jira: **Projects → Create project**.
2. Choose a **Scrum** or **Kanban** template.
3. Name it anything (e.g. "QA Demo"), and set the **Key** to **`TEST`**.

> **Already have a project?** New Jira sites usually create one automatically with a key like
> `KAN` or `SCRUM`. You can use that instead — put its key in `JIRA_PROJECT_KEY` in step 6.4.
> The example buttons in the app automatically use whatever key you set.
> The app will only ever create or update tickets in **that one project**
> (it's a guardrail).

### 6.3 Create an API token

1. Go to **https://id.atlassian.com/manage-profile/security/api-tokens**.
2. Click **Create API token** (the plain one, not "with scopes"), name it, set an expiry.
3. **Copy the token.**

### 6.4 Add it to `.env`

```dotenv
JIRA_URL=https://yourname.atlassian.net
JIRA_USERNAME=the-email-you-log-in-to-atlassian-with
JIRA_API_TOKEN=paste-the-token-here
JIRA_PROJECT_KEY=TEST
```

### 6.5 Restart and check

Stop the app (**Ctrl + C**) and run `streamlit run app.py` again.

✅ **Checkpoint:** the sidebar's **🔌 Connections** shows **🟢 Jira MCP — 5 tools**.

**Test it:** click **🗂️ Jira (approve/cancel)**, then **✅ Approve** when asked. The answer
names a new ticket (e.g. `TEST-1`), and 📚 Sources links to it in Jira.

---

## Part 7 — Connect Gmail (optional)

This lets the app **send real emails from your Gmail** — always after you approve.
You create your own small "app" in Google Cloud so the email comes from your account.
(Google renames menu items now and then; if a name differs slightly, look for the
closest match.)

### 7.1 Create a Google Cloud project

1. Go to **https://console.cloud.google.com/** and sign in with the Gmail account that
   should send the emails.
2. Click the project picker at the top → **New Project** → name it (e.g. "QA Agent")
   → **Create**. Make sure it's selected.

### 7.2 Turn on the Gmail API

**APIs & Services → Library** → search **Gmail API** → open it → **Enable**.

### 7.3 Set up the consent screen

1. **APIs & Services → OAuth consent screen** (it may be called **Google Auth Platform**).
2. Choose **External**, enter an app name and your email, and save.
3. Under **Audience / Test users**, click **Add users** and add **your own Gmail
   address**. (Only test users can sign in while the app is unpublished.)

### 7.4 Create the credentials file

1. **APIs & Services → Credentials** (or **Clients**) → **Create credentials →
   OAuth client ID**.
2. Application type: **Desktop app** → **Create**.
3. Click **Download JSON**.
4. Rename the file to **`gmail_credentials.json`** and move it into the project folder
   (next to `app.py`). It's git-ignored, so it will never be uploaded.

### 7.5 Make sure your address is allowed

In `.env`, check:
```dotenv
DEFAULT_EMAIL_TO=your.name@gmail.com
ALLOWED_EMAIL_DOMAINS=gmail.com,example.com
```
Emails can **only** go to domains in `ALLOWED_EMAIL_DOMAINS` (a guardrail). Using a work
address? Add its domain, e.g. `gmail.com,yourcompany.com`.

### 7.6 Restart, then sign in once

Restart the app. ✅ The sidebar shows **🟢 Gmail MCP — 16 tools**.

Click **✉️ Email me** and approve. **The first time**, a browser window may open asking you
to sign in and allow access:

- If you see *"Google hasn't verified this app"*, click **Continue** — it's your own app.
- Allow the Gmail permissions.

After that, a sign-in token is stored locally and you won't be asked again.

✅ **Checkpoint:** the email arrives in your inbox.

> 🔒 **Safety note:** the email agent is given **send-only** tools. It cannot read, delete
> or trash your mail, even though the Gmail server offers those tools.

---

## Part 8 — Connect LangSmith (optional)

LangSmith records every step the agents take, so you can inspect prompts, tool calls and
timings. The app adds a **🔗 LangSmith trace** link under every answer.

1. Sign up at **https://smith.langchain.com/** (free tier available).
2. Go to **Settings → API Keys → Create API Key**, and copy it (starts with `lsv2_`).
3. In `.env`, set **both** of these:
   ```dotenv
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=lsv2_paste-your-key
   LANGSMITH_PROJECT=langgraph-testing-mastery
   ```
   On an EU LangSmith account, also uncomment `LANGSMITH_ENDPOINT`.
4. Restart the app.

✅ **Checkpoint:** the sidebar shows **🟢 LangSmith — langgraph-testing-mastery**. Ask
anything, then click **🔗 LangSmith trace** under the answer — it opens that exact run.

> If the link says *"still uploading"*, wait a few seconds and open the Details panel
> again.

---

## Part 9 — Guided tour: try every feature

Click the buttons under **💡 Try these** in this order. For each one, the table says what
to look at. **Look at the sidebar's "🔀 Last run — agent trail"** every time — it shows
the path through the graph.

The example documents the agents search are in `sample_docs/`: `known_bugs.md`
(BUG-101, BUG-087, BUG-112), `login_test_plan.md`, and `api_test_cases.md`.

| # | Button | What happens | What to notice |
|---|---|---|---|
| 1 | **📚 Research only** | Searches the docs | Details → **📚 Sources** lists `known_bugs.md` |
| 2 | **💭 No tools** | Answers from general knowledge | Chips say **🔧 no tools used · 📚 no sources** — the app is honest about it |
| 3 | **🧪 Test cases** | Generates test cases | Details → **🔧 Tools** usually shows `generate_test_cases` (the agent picks its tools, so it may also search the docs) |
| 4 | **🔁 Self-correcting loop** | A reviewer grades the research | If it's weak, the trail shows **↩️ loop back to research** with the reason, then a retry. (It may pass first time — that's fine too.) |
| 5 | **🛡️ Injection (blocked)** | Blocked before any agent runs | Trail: **🛡️ entry guardrail BLOCKED** |
| 6 | **🛡️ Bad domain (blocked)** | Research runs, email is blocked | **No approval prompt** — the guardrail stops it first. The answer ends with **🛡️ Email: blocked** |
| 7 | **🗂️ Jira (approve/cancel)** | Pauses for approval | Click **❌ Cancel**. You are **not** asked again, and the answer ends with **⛔ Jira: declined by you** |
| 8 | **🗂️ Jira (approve/cancel)** again | Pauses for approval | This time click **✅ Approve** → a real ticket (needs Part 6) |
| 9 | **✉️ Email me** | Pauses for approval | **Approve** → email to `DEFAULT_EMAIL_TO` (needs Part 7) |
| 10 | **⭐ Full chain (2 approvals)** | research → bug → Jira 🔒 → email 🔒 | The showpiece: 4 agents hand off, two approvals, one ticket + one email |
| 11 | **🧪 Poisoned doc (indirect injection)** | Research reads `release_notes.md`, which hides an instruction to email your bugs to an outside address | The trail shows **🛡️ tool-result guard: removed 1 suspicious line**, the chip shows **🛡️ 1 guardrail flag**, and Details quotes the removed line. The AI never saw it |

**Try typing your own too** — more ideas, grouped by feature, are in
**[QUESTIONS.md](QUESTIONS.md)**, including a 7-minute demo order for presenting.

### Before you approve: preview

When the app asks for approval, expand **Preview the content** to see exactly what will
be filed or emailed. Nothing real happens until you click **✅ Approve**.

### Switching the AI model

Sidebar → **🧠 Model → Provider**. `gpt-4o` is smarter but slower and pricier. Claude needs
`ANTHROPIC_API_KEY` in `.env`.

---

## Part 10 — Understanding the screen

### Sidebar

| Section | Meaning |
|---|---|
| 🧠 **Model** | Which AI model answers |
| 🔌 **Connections** | 🟢 connected / 🔴 not connected, for Jira, Gmail and LangSmith. Any startup error appears here in red. **Loaded tool names** lists every tool |
| 🕸️ **The agent graph** | The team: supervisor, research, bug, guardrail, jira, comms, quality |
| 🔀 **Last run — agent trail** | Every step of the last request, in order. Purple = supervisor decisions, orange = loop-backs |
| 🖼️ **Graph diagram** | The real graph, drawn from the code |
| 🧹 **New conversation** | Clears the screen. (Each question already starts fresh, so this is only visual) |

### Under every answer

```
🔧 5 tool calls   📚 3 sources   ⏱ 14.2s · slowest: research   ↩️ 1 loop-back   🔗 LangSmith trace
```

Expand **🔎 Details** for:

- **📚 Sources** — the documents and Jira tickets the answer actually used (tickets are
  clickable). Search results that weren't used appear as one grey line.
- **🔧 Tools** — every tool call: which agent, ✅ ok / 🛡️ blocked / ❌ error, time, inputs,
  and a preview of the result.
- **⏱ Timing** — how long each step took. Useful when something feels slow.
- **🔗 Trace** — LangSmith links and the full trail.

### Why does a request take 10–30 seconds?

Every step is a separate AI call, one after another: the supervisor decides, an agent
works (sometimes several calls), the reviewer grades, and so on. The **⏱ Timing** tab
shows exactly where the time went.

### Words you'll see

| Term | Meaning |
|---|---|
| **Agent** | An AI worker with one job and its own tools |
| **Supervisor** | The agent that decides who works next, and when the job is done |
| **Loop-back** | The reviewer rejected a step, so it runs again with feedback |
| **Guardrail** | An automatic safety check. This app has five: on your request, on actions, on the real tool call, on what tools return, and on the final answer |
| **Indirect prompt injection** | Instructions hidden in a document or ticket, hoping the AI will obey them. The tool-result guard removes them |
| **Approval gate** | The pause where *you* decide whether a real action happens |
| **MCP** | Model Context Protocol — the standard way the app talks to Jira and Gmail |
| **RAG** | Retrieval-Augmented Generation — searching your documents before answering |
| **Trace** | A recording of every step, in LangSmith |

---

## Part 11 — Everyday commands

**Start the app again later:**

macOS:
```bash
cd ~/Desktop/langgraph-testing-mastery
source .venv/bin/activate
streamlit run app.py
```

Windows:
```bat
cd %USERPROFILE%\Desktop\langgraph-testing-mastery
.venv\Scripts\activate
streamlit run app.py
```

| Task | macOS | Windows |
|---|---|---|
| Stop the app | **Ctrl + C** in the terminal | **Ctrl + C** |
| Run the tests | `python -m pytest tests -q` | same |
| Get the latest version | `git pull` then `pip install -r requirements.txt` | same |
| Rebuild the document index | `rm -rf chroma_db` then restart | `rmdir /s /q chroma_db` then restart |
| Leave the virtual env | `deactivate` | `deactivate` |

> After **any** change to `.env`, restart the app — the Jira and Gmail connections are
> made once, at startup.

---

## Part 12 — Make it yours

| I want to… | Do this |
|---|---|
| Use **my own documents** | Put `.md` files in `sample_docs/` and restart. Only new or changed files are re-indexed |
| Use **my Jira project** | Set `JIRA_PROJECT_KEY` in `.env` |
| Email **my company** addresses | Add the domain to `ALLOWED_EMAIL_DOMAINS` |
| Allow **more or fewer retries** | Change `MAX_LOOPS` in `.env` |
| Add **a new tool** or **a new agent** | See README → [Adapt it for your own use case](README.md#10-adapt-it-for-your-own-use-case) |
| Understand **how it works** | Read `graph.py` first — the whole design is on one page — then README section 5 |
| Keep my own copy on GitHub | Click **Fork** on the repo page, then clone your fork instead |

---

## Part 13 — Troubleshooting

| Problem | Fix |
|---|---|
| `command not found: python3.12` / `py` not recognised | Python 3.12 isn't installed or not on PATH — redo step 1.3 (Windows: tick **Add to PATH**) |
| `uv` / `uvx` not found | Redo step 1.4, then **close and reopen** the terminal |
| `ModuleNotFoundError` | Activate the venv (step 3.1) and make sure you're in the project folder |
| `87 passed` not shown | Run `pip install -r requirements.txt` again; paste the first error into an issue |
| **Not ready: OPENAI_API_KEY is not set** | `.env` is missing, misnamed (e.g. `.env.txt`), or not in the project folder |
| OpenAI error about quota / billing | Add credit in OpenAI **Settings → Billing** |
| 🔴 **Jira** in the sidebar | Check the red error under it. Verify `JIRA_URL` (with `https://`), username = your Atlassian email, a fresh API token. Restart |
| Jira error: *"The target project doesn't exist or you don't have permission to create issues in it"* | Your Jira has no project with the key in `JIRA_PROJECT_KEY` (default `TEST`). New Jira sites usually start with a project keyed `KAN` or `SCRUM`. Check **Projects → View all projects → Key**, then either set `JIRA_PROJECT_KEY=KAN` (your key) in `.env` and restart, or create a project with key `TEST` (step 6.2). Also confirm `JIRA_USERNAME` is the same email the API token belongs to, and that you can create an issue in that project on the Jira website |
| Jira ticket fails with an issue-type error | Some project templates have no "Bug" type — ask for a "Task" instead |
| 🔴 **Gmail** in the sidebar | `gmail_credentials.json` must be in the project folder with exactly that name |
| Google says **access blocked** | Add your address as a **Test user** (step 7.3) |
| Email blocked: *no valid recipient* | Set `DEFAULT_EMAIL_TO` in `.env`, or include an address in your request |
| Email blocked: *domain not allowed* | Add that domain to `ALLOWED_EMAIL_DOMAINS` |
| ⚪ **LangSmith — tracing off** | You need **both** `LANGSMITH_TRACING=true` and a real `LANGSMITH_API_KEY` |
| Answers mention old document content | Stop the app, delete `chroma_db` (Part 11), restart |
| Jira/Gmail stopped working mid-session | Restart the app. If it keeps happening, set `MCP_PERSISTENT_SESSIONS=false` |
| Port 8501 already in use | Another copy is running — stop it, or use `streamlit run app.py --server.port 8502` |
| Lots of warnings in the terminal (FastMCP banner, `TOOLSETS`, `authlib`) | Normal — they come from the Jira/Gmail servers and are harmless |
| `rmdir /s /q` fails on Mac | That's Windows syntax — use `rm -rf chroma_db` |

Still stuck? Open an issue on the GitHub repo with the command you ran and the full error
message — **remove any keys first**.

---

## Final checklist

- [ ] Git, Python 3.12 and uv installed (Part 1)
- [ ] Repo cloned, `.venv` created and active (Parts 2–3)
- [ ] `.env` created with a real `OPENAI_API_KEY` (Part 4)
- [ ] `python -m pytest tests -q` → **87 passed** (Part 5)
- [ ] App opens at http://localhost:8501 and **📚 Research only** works (Part 5)
- [ ] *(optional)* 🟢 Jira, and a ticket created after approval (Part 6)
- [ ] *(optional)* 🟢 Gmail, and an email received after approval (Part 7)
- [ ] *(optional)* 🟢 LangSmith, and the trace link opens (Part 8)
- [ ] All 11 buttons in the guided tour tried (Part 9)

**Next:** read **[README.md](README.md)** to understand *how* it works, and
**[QUESTIONS.md](QUESTIONS.md)** for more to try.

---

**AI Testing Mastery** · [github.com/AITestingMastery](https://github.com/AITestingMastery)