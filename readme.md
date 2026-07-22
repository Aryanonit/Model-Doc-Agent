# 🤖 ModelDocAgent: The AI Architecture Mapper

## 📖 What is this project?
Welcome to **ModelDocAgent**. If you are reading this, you are looking at a fully automated, asynchronous Google Chat bot that reads minds (well, codebases). 

**The Goal:** A developer pastes a GitHub repository link into a Google Chat room. The bot instantly downloads the code, reads it using Google's Gemini AI, and replies with a highly formatted UI card containing a 20-point architectural specification and a live Mermaid.js system diagram. 

We built this entirely from scratch, transforming a chaotic web of terminal commands into a sleek, professional, unified codebase.

---

## 🏗️ How the System Actually Works
The pipeline is a multi-stage relay race across several APIs:
1. **The Hook:** Google Chat sends a webhook payload to our local server via an Ngrok secure tunnel.
2. **The Handshake:** Our Flask server instantly acknowledges the message to prevent Google from timing out.
3. **The Fetch:** An `AuthenticatedRepoIngestor` class uses a GitHub Personal Access Token to quietly read the repository's files (ignoring junk like `node_modules` or `.venv`).
4. **The Brain:** The compiled codebase is handed to `gemini-2.5-flash` using LangChain and Pydantic schemas to ensure the AI's output is rigidly formatted as JSON (Specs + Mermaid code).
5. **The Artist:** The Mermaid code is cleaned, base64-encoded, and turned into a live image URL via `mermaid.ink`.
6. **The Delivery:** A background Python thread authenticates with Google Cloud using a Service Account Key, builds a beautiful `CardsV2` UI payload, and injects the final document back into the chat.

---

## 🧗‍♂️ The Journey & The Great Bug Log
Building this wasn't a straight line. We hit several massive bottlenecks, navigated undocumented API behaviors, and solved them one by one. Here is the war room log of every trap we fell into and how we escaped:

### 🐛 Bug 1: The Ngrok `.dev` vs `.app` Trap
* **What Happened:** We told Google Cloud to send webhooks to `ngrok-free.app`, but Ngrok assigned us a `ngrok-free.dev` domain. Google was sending messages into a black hole.
* **The Fix:** We aligned the exact terminal forwarding URL with the Google Cloud Console configuration.

### 🐛 Bug 2: The Flask Trailing Slash Redirect
* **What Happened:** Flask is notoriously strict. Our code said `@app.route('/')`. When Google Chat sent a request to the root URL *without* the slash, Flask threw a 308 Redirect, dropping the payload entirely.
* **The Fix:** We ensured the endpoint URL configured in Google Cloud exactly matched Flask's routing expectations.

### 🐛 Bug 3: The 30-Second Google Guillotine (The Hardest Battle)
* **What Happened:** Google Chat requires bots to reply within exactly 30 seconds. Our AI pipeline (fetching code, reading it, generating images) took 35-40 seconds. The server was returning a `200 OK` success code, but Google Chat was hanging up early and displaying `"DocAgent not responding"`.
* **The Fix:** We completely re-architected the app. We added Python `threading` to instantly reply to the webhook with `{"text": "⏳ I'm on it..."}` and shifted the heavy AI processing to a background thread.

### 🐛 Bug 4: Flask's Aggressive Thread Killer
* **What Happened:** Even after adding background threads, the bot still crashed silently.
* **The Fix:** We realized Flask's default `debug=True` development mode actively murders background threads the second the main HTTP request finishes to save memory. We bypassed this by switching to `debug=False, threaded=True`.

### 🐛 Bug 5: The Wrong Key (OAuth vs. Service Account)
* **What Happened:** For the background thread to reply to Google Chat, it needs a credential file. We initially tried to use an OAuth Client ID JSON, and then an expired Qwiklabs sandbox JSON. The script crashed because it lacked server-to-server permissions.
* **The Fix:** We generated a proper **Service Account Key** (`credentials.json`) from Google Cloud IAM, giving the bot a "VIP Robot Passport" to act on its own.

### 🐛 Bug 6: Terminal Chaos & Environment Ghosting
* **What Happened:** We were running the project using raw `nano` in multiple scattered Linux terminal windows. Keys were being lost between terminal sessions, and Python couldn't find installed modules.
* **The Fix:** "Antigravity". We migrated the entire project into a unified Visual Studio Code workspace. We created an isolated virtual environment (`venv`), centralized our keys into a hidden `.env` file using `python-dotenv`, and combined all modular scripts into a single, master `app.py` engine.

---

## 🗂️ The Current Workspace State
Everything is now unified in the `ModelDocAgent` directory:
* `venv/` $\rightarrow$ The isolated Python environment.
* `.env` $\rightarrow$ Safe storage for `GEMINI_KEY` and `GITHUB_PAT`.
* `credentials.json` $\rightarrow$ The Google Cloud Service Account key.
* `app.py` $\rightarrow$ The monolithic master script containing the server, ingestor, orchestrator, and chat payload logic.

## 🚀 How to Launch
1. Open the folder in VS Code.
2. Ensure the virtual environment is active: `source venv/bin/activate`
3. Run the public tunnel in a background terminal: `ngrok http 8080 &`
4. Boot the engine: `python app.py`     




# 🤖 ModelDocAgent: AI Architecture Mapper

## 📖 Project Overview
ModelDocAgent is an automated Google Chat bot designed to instantly document and map software architectures. When a user drops a GitHub repository URL into a Google Chat space, the bot ingests the entire codebase, analyzes it using Google's Gemini 2.5 Flash AI, and replies with a highly formatted UI card containing a 20-point technical specification and a live Mermaid.js architecture diagram.

## 🛠️ Tech Stack & Architecture
* **Core Engine:** Python 3.12
* **Web Server:** Flask (running locally on port 8080)
* **Tunneling:** Ngrok (exposes local Flask server to the public internet for Google webhooks)
* **AI Model:** Google GenAI SDK (`gemini-2.5-flash`)
* **Orchestration & Tooling:** LangChain Core, Pydantic (for rigid JSON schema enforcement)
* **Platform Integrations:** * Google Chat API (Webhooks, Background Messaging, CardsV2 UI)
  * GitHub API (Authenticated codebase ingestion)

## 🗺️ The Pipeline (How It Works)
1. **The Trigger:** A user sends a GitHub link to the bot in Google Chat.
2. **The Catch:** Google Cloud routes the message through the Ngrok tunnel to the local Flask server (`app.py`).
3. **The Acknowledgment:** To beat Google Chat's strict 30-second timeout rule, Flask immediately replies with a "⏳ I'm on it!" text and spins up a background thread.
4. **The Ingestion:** The `AuthenticatedRepoIngestor` class uses a GitHub Personal Access Token (PAT) to bypass rate limits, clone the repo structure, filter out junk files (`node_modules`, `.venv`), and compile the raw code into a context buffer.
5. **The Brain:** The `ProductionDocOrchestrator` feeds the codebase to Gemini. Using a Pydantic schema, it forces Gemini to output exactly two things: a 20-point bulleted list, and raw Mermaid.js syntax.
6. **The Renderer:** The `open_source_diagram_generator_tool` cleans the Mermaid code, encodes it to Base64, and generates a live image link via `mermaid.ink`.
7. **The Delivery:** The background thread authenticates using a Google Cloud Service Account JSON key, builds a structured `CardsV2` UI payload, and pushes the final document and image directly back into the Google Chat space.

## 📅 Project Evolution & Milestones (Our Journey)
* **Phase 1: Conceptualization & Brainstorming:** Outlining the goal of reading a repo and outputting diagrams.
* **Phase 2: AI Tooling Setup:** Building the LangChain tools, setting up Pydantic schemas, and fighting with Mermaid syntax to ensure it renders flawlessly.
* **Phase 3: The GitHub Connector:** Writing the HTTPX logic to securely fetch private/public repos without filling up local hard drive space.
* **Phase 4: Google Cloud Wiring:** Setting up the Google Chat API, creating Service Accounts, and pointing the endpoint to Ngrok.
* **Phase 5: The Timeout Battle:** Discovering Google Chat's 30-second guillotine and rewriting the Flask app to use asynchronous background threading (`threading.Thread`).
* **Phase 6: The Great Consolidation:** Moving out of messy terminal windows (`nano`) into Visual Studio Code, setting up a proper Python Virtual Environment (`venv`), and securing keys in a `.env` file.

## 📁 Current Workspace Structure
Everything is now unified under one roof:
```text
ModelDocAgent/
├── venv/                  # The isolated Python sandbox
├── .env                   # Hidden file containing API keys (Gemini, GitHub, etc.)
├── credentials.json       # Google Cloud Service Account robot passport
├── app.py                 # The monolithic master engine (Flask + AI + GitHub)
└── README.md              # This documentation file