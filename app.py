import os
import re
import json
import base64
import httpx
import threading
import sys
import time
import zlib
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from google import genai
from google.genai import types
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials as UserCredentials
from google.auth.transport.requests import Request as GoogleAuthRequest
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

app = Flask(__name__)

# =====================================================================
# 1. CORE PIPELINE COMPONENTS
# =====================================================================
class SpecSchema(BaseModel):
    document_content: list[str] = Field(description="A highly detailed 20-point corporate architecture breakdown.")
    diagram_code: str = Field(description="Flawless Mermaid.js architecture diagram syntax mapping the system.")

class DocSection(BaseModel):
    heading: str = Field(description="A short, clear section title, e.g. 'Data Ingestion Pipeline'.")
    body: str = Field(description="2-4 full sentences of real explanatory prose for this section -- not a bullet fragment.")

class FileEntry(BaseModel):
    path: str = Field(description="Relative file path exactly as it appears in the repo, e.g. 'app.py' or 'cta_features/cta_master.py'.")
    description: str = Field(description="1-2 sentences: what this specific file does and why it matters to the system.")

class DirectoryGroup(BaseModel):
    group_name: str = Field(description="A short label for this group, e.g. 'Core Application & Server' or the directory name itself.")
    files: list[FileEntry]

class DeepDocSchema(BaseModel):
    project_name: str = Field(description="The project/system's name, inferred from the repo (folder name, README title, or package name).")
    overview: str = Field(description="4-6 sentences: what this codebase is, its primary purpose, and its core technical approach.")
    execution_flow: list[str] = Field(description="4-8 ordered steps describing the end-to-end execution flow, each a full sentence or two, in the actual order things happen.")
    sections: list[DocSection] = Field(description="6-10 thematic sections (e.g. Data Layer, API Layer, Feature Pipeline) each with a heading and explanatory paragraph.")
    file_breakdown: list[DirectoryGroup] = Field(description="Every meaningfully analyzed file, grouped by its directory/module, each with a short description of what it does.")
    use_cases: list[str] = Field(description="3-6 real-world use cases this system enables for the people who'd use it.")

def sanitize_mermaid(code: str) -> str:
    """Cleans up LLM syntax layout anomalies and guarantees valid Mermaid structure spacing."""
    # Strip any markdown code block wrappers if included
    code = code.replace("```mermaid", "").replace("```", "").strip()

    # Force an explicit newline if the model merged flowchart layout and subgraph together
    code = re.sub(r'(flowchart\s+[A-Z]{2})\s+(subgraph)', r'\1\n\2', code, flags=re.IGNORECASE)

    # Guarantee newlines before structural boundary keyword tags
    code = re.sub(r'(\S)\s+(subgraph|end\b)', r'\1\n\2', code, flags=re.IGNORECASE)

    lines = code.splitlines()
    cleaned = []
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        # Strip trailing hanging link symbols with missing destinations
        if re.search(r'(-->|--x|--o|-\.->)\s*$', line_str):
            continue
        cleaned.append(line)

    return "\n".join(cleaned)

DOC_SCOPES = ['https://www.googleapis.com/auth/documents', 'https://www.googleapis.com/auth/drive.file']
CHAT_SCOPES = ['https://www.googleapis.com/auth/chat.bot']

_logo_url_cache = {}

def get_logo_url(credentials, logo_path: str = None):
    """Uploads a local logo image file to the user's Drive once (cached in-process
    afterward) and returns a URL the Docs API can fetch for header/cover images.
    Set LOGO_IMAGE_PATH in .env to a local image file path to enable this --
    if unset or the file doesn't exist, returns None and the logo slot is skipped."""
    logo_path = logo_path or os.environ.get("LOGO_IMAGE_PATH")
    if not logo_path or not os.path.exists(logo_path):
        return None
    if logo_path in _logo_url_cache:
        return _logo_url_cache[logo_path]

    drive_service = build('drive', 'v3', credentials=credentials)
    media = MediaFileUpload(logo_path, resumable=False)
    file = drive_service.files().create(
        body={'name': os.path.basename(logo_path)},
        media_body=media,
        fields='id'
    ).execute()
    file_id = file.get('id')
    drive_service.permissions().create(
        fileId=file_id, body={'type': 'anyone', 'role': 'reader'}
    ).execute()

    url = f"https://drive.google.com/uc?export=view&id={file_id}"
    _logo_url_cache[logo_path] = url
    return url

def is_valid_image_url(url: str, timeout: float = 8.0) -> bool:
    """Confirms a URL actually returns image bytes before trusting it in a Doc or Chat card,
    so a broken diagram render degrades gracefully instead of causing a downstream
    'problem retrieving the image' failure."""
    if not url:
        return False
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
        content_type = resp.headers.get("content-type", "")
        return resp.status_code == 200 and content_type.startswith("image/")
    except Exception:
        return False

def get_user_credentials(token_path: str = 'token.json'):
    """Loads your personal OAuth token (from authorize_google.py) so created Docs
    are owned by you, not the service account -- avoiding the 0-byte service
    account storage quota entirely. Auto-refreshes if expired."""
    if not os.path.exists(token_path):
        raise Exception(f"{token_path} not found. Run authorize_google.py once to create it.")
    creds = UserCredentials.from_authorized_user_file(token_path, DOC_SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(GoogleAuthRequest())
        with open(token_path, 'w') as f:
            f.write(creds.to_json())
    return creds

FIGTREE = 'Figtree'
HIGHLIGHT_YELLOW = {'color': {'rgbColor': {'red': 1.0, 'green': 0.976, 'blue': 0.0}}}
RULE_GRAY = {'color': {'rgbColor': {'red': 0.6, 'green': 0.6, 'blue': 0.6}}}

def create_google_doc(credentials, title: str, sections: list, diagram_url: str,
                       subtitle: str = None, author_name: str = None,
                       parent_folder_id: str = None, logo_path: str = None,
                       overview: str = None, execution_flow: list = None,
                       file_breakdown: list = None, use_cases: list = None,
                       github_base_url: str = None) -> str:
    """Creates a formatted Google Doc: cover page (title/subtitle/author/date with
    yellow highlight + horizontal rule, matching the company template) followed by
    content pages using Heading 2 / Body text styles in Figtree, round bullets,
    and a repeating header logo + footer across every page."""
    docs_service = build('docs', 'v1', credentials=credentials)
    drive_service = build('drive', 'v3', credentials=credentials)

    file_metadata = {'name': title, 'mimeType': 'application/vnd.google-apps.document'}
    if parent_folder_id:
        file_metadata['parents'] = [parent_folder_id]
    file = drive_service.files().create(body=file_metadata, fields='id').execute()
    doc_id = file.get('id')

    subtitle = subtitle or "Architecture Blueprint"
    author_name = author_name or "DocAgent"
    date_str = datetime.now().strftime("%d %B %Y")

    # ---------- 1. COVER PAGE TEXT ----------
    # Blank lines push the title block down the page, matching the template's layout.
    cover_lead_in = "\n" * 8
    cover_parts = [cover_lead_in, f"{title}\n", f"{subtitle}\n", f"{author_name}\n", f"{date_str}\n\n"]
    cover_text = "".join(cover_parts)

    docs_service.documents().batchUpdate(
        documentId=doc_id,
        body={'requests': [{'insertText': {'location': {'index': 1}, 'text': cover_text}}]}
    ).execute()

    # Compute offsets within cover_text for each styled line
    lead_len = len(cover_lead_in)
    title_start = lead_len
    title_end = title_start + len(title)
    subtitle_start = title_end + 1
    subtitle_end = subtitle_start + len(subtitle)
    author_start = subtitle_end + 1
    author_end = author_start + len(author_name)
    date_start = author_end + 1
    date_end = date_start + len(date_str)

    cover_style_requests = [
        # Title: large, bold, yellow-highlighted
        {'updateTextStyle': {
            'range': {'startIndex': 1 + title_start, 'endIndex': 1 + title_end},
            'textStyle': {
                'bold': False, 'fontSize': {'magnitude': 32, 'unit': 'PT'},
                'weightedFontFamily': {'fontFamily': FIGTREE, 'weight': 400},
                'backgroundColor': HIGHLIGHT_YELLOW
            },
            'fields': 'fontSize,weightedFontFamily,backgroundColor'
        }},
        # Subtitle: bold, smaller, yellow-highlighted
        {'updateTextStyle': {
            'range': {'startIndex': 1 + subtitle_start, 'endIndex': 1 + subtitle_end},
            'textStyle': {
                'bold': True, 'fontSize': {'magnitude': 15, 'unit': 'PT'},
                'weightedFontFamily': {'fontFamily': FIGTREE, 'weight': 600},
                'backgroundColor': HIGHLIGHT_YELLOW
            },
            'fields': 'bold,fontSize,weightedFontFamily,backgroundColor'
        }},
        # Author + date: small, yellow-highlighted
        {'updateTextStyle': {
            'range': {'startIndex': 1 + author_start, 'endIndex': 1 + date_end},
            'textStyle': {
                'fontSize': {'magnitude': 11, 'unit': 'PT'},
                'weightedFontFamily': {'fontFamily': FIGTREE, 'weight': 400},
                'backgroundColor': HIGHLIGHT_YELLOW
            },
            'fields': 'fontSize,weightedFontFamily,backgroundColor'
        }},
        # Horizontal rule above the title (border on the blank line just before it)
        {'updateParagraphStyle': {
            'range': {'startIndex': 1 + title_start - 1, 'endIndex': 1 + title_start},
            'paragraphStyle': {'borderBottom': {
                'color': RULE_GRAY, 'width': {'magnitude': 1, 'unit': 'PT'},
                'padding': {'magnitude': 4, 'unit': 'PT'}, 'dashStyle': 'SOLID'
            }},
            'fields': 'borderBottom'
        }},
        # Horizontal rule under the subtitle
        {'updateParagraphStyle': {
            'range': {'startIndex': 1 + subtitle_start, 'endIndex': 1 + subtitle_end},
            'paragraphStyle': {'borderBottom': {
                'color': RULE_GRAY, 'width': {'magnitude': 1, 'unit': 'PT'},
                'padding': {'magnitude': 4, 'unit': 'PT'}, 'dashStyle': 'SOLID'
            }},
            'fields': 'borderBottom'
        }},
    ]
    docs_service.documents().batchUpdate(documentId=doc_id, body={'requests': cover_style_requests}).execute()

    # ---------- 2. PAGE BREAK INTO CONTENT ----------
    cover_end_index = 1 + len(cover_text)
    docs_service.documents().batchUpdate(
        documentId=doc_id,
        body={'requests': [{'insertPageBreak': {'location': {'index': cover_end_index}}}]}
    ).execute()
    content_start_index = cover_end_index + 1

    # ---------- 3. CONTENT: build as tagged blocks so each part (headings, numbered
    # steps, hyperlinked file paths, bullets) can get its own formatting pass afterward ----------
    blocks = []  # each: {'type': ..., 'start': int, 'end': int, 'link_start': int, 'link_end': int}
    text_parts = []
    cursor = 0

    def add_block(text, block_type, is_paragraph_end=True):
        nonlocal cursor
        start = cursor
        text_parts.append(text)
        cursor += len(text)
        blocks.append({'type': block_type, 'start': start, 'end': cursor - (1 if is_paragraph_end else 0)})

    if overview:
        add_block("Overview\n", 'h1')
        add_block(f"{overview}\n\n", 'body')

    if execution_flow:
        add_block("End-to-End Execution Flow\n", 'h1')
        flow_start_block_index = len(blocks)
        for step in execution_flow:
            add_block(f"{step.strip()}\n", 'numbered_item')
        add_block("\n", 'spacer', is_paragraph_end=False)

    for section in sections:
        heading = (section.get('heading') or '').strip()
        body = (section.get('body') or '').strip()
        if not heading and not body:
            continue
        add_block(f"{heading}\n", 'h2')
        add_block(f"{body}\n\n", 'body')

    if file_breakdown:
        add_block("File-by-File Breakdown\n", 'h1')
        for group in file_breakdown:
            group_name = (group.get('group_name') or '').strip()
            if group_name:
                add_block(f"{group_name}\n", 'h2')
            for f in group.get('files', []):
                path = (f.get('path') or '').strip()
                desc = (f.get('description') or '').strip()
                line = f"{path} — {desc}\n"
                start = cursor
                text_parts.append(line)
                cursor += len(line)
                blocks.append({
                    'type': 'file_item', 'start': start, 'end': cursor - 1,
                    'link_start': start, 'link_end': start + len(path),
                    'url': f"{github_base_url}{path}" if github_base_url else None
                })
        add_block("\n", 'spacer', is_paragraph_end=False)

    if use_cases:
        add_block("What This Enables\n", 'h1')
        for uc in use_cases:
            add_block(f"{uc.strip()}\n", 'bullet_item')

    content_text = "".join(text_parts)
    docs_service.documents().batchUpdate(
        documentId=doc_id,
        body={'requests': [{'insertText': {'location': {'index': content_start_index}, 'text': content_text}}]}
    ).execute()

    # Base styling: Figtree body text, 12pt, 1.5 line spacing, across all content
    content_style_requests = [
        {'updateTextStyle': {
            'range': {'startIndex': content_start_index, 'endIndex': content_start_index + len(content_text)},
            'textStyle': {'fontSize': {'magnitude': 12, 'unit': 'PT'}, 'weightedFontFamily': {'fontFamily': FIGTREE, 'weight': 400}},
            'fields': 'fontSize,weightedFontFamily'
        }},
        {'updateParagraphStyle': {
            'range': {'startIndex': content_start_index, 'endIndex': content_start_index + len(content_text)},
            'paragraphStyle': {'lineSpacing': 150},
            'fields': 'lineSpacing'
        }},
    ]

    for block in blocks:
        abs_start = content_start_index + block['start']
        abs_end = content_start_index + block['end']
        if block['type'] == 'h1':
            content_style_requests.append({'updateTextStyle': {
                'range': {'startIndex': abs_start, 'endIndex': abs_end},
                'textStyle': {'bold': True, 'fontSize': {'magnitude': 16, 'unit': 'PT'}, 'weightedFontFamily': {'fontFamily': FIGTREE, 'weight': 700}},
                'fields': 'bold,fontSize,weightedFontFamily'
            }})
        elif block['type'] == 'h2':
            content_style_requests.append({'updateTextStyle': {
                'range': {'startIndex': abs_start, 'endIndex': abs_end},
                'textStyle': {'bold': True, 'fontSize': {'magnitude': 14, 'unit': 'PT'}, 'weightedFontFamily': {'fontFamily': FIGTREE, 'weight': 600}},
                'fields': 'bold,fontSize,weightedFontFamily'
            }})
        elif block['type'] == 'file_item' and block.get('url'):
            content_style_requests.append({'updateTextStyle': {
                'range': {'startIndex': content_start_index + block['link_start'], 'endIndex': content_start_index + block['link_end']},
                'textStyle': {'link': {'url': block['url']}, 'weightedFontFamily': {'fontFamily': 'Courier New', 'weight': 400}, 'fontSize': {'magnitude': 10.5, 'unit': 'PT'}},
                'fields': 'link,weightedFontFamily,fontSize'
            }})
    docs_service.documents().batchUpdate(documentId=doc_id, body={'requests': content_style_requests}).execute()

    # Bullet/numbering pass -- group contiguous same-type items into single ranges
    def contiguous_ranges(item_type):
        items = [b for b in blocks if b['type'] == item_type]
        ranges = []
        for b in items:
            if ranges and b['start'] == ranges[-1][1]:
                ranges[-1] = (ranges[-1][0], b['end'] + 1)
            else:
                ranges.append((b['start'], b['end'] + 1))
        return ranges

    list_requests = []
    for (r_start, r_end) in contiguous_ranges('numbered_item'):
        list_requests.append({'createParagraphBullets': {
            'range': {'startIndex': content_start_index + r_start, 'endIndex': content_start_index + r_end},
            'bulletPreset': 'NUMBERED_DECIMAL_ALPHA_ROMAN'
        }})
    for (r_start, r_end) in contiguous_ranges('bullet_item'):
        list_requests.append({'createParagraphBullets': {
            'range': {'startIndex': content_start_index + r_start, 'endIndex': content_start_index + r_end},
            'bulletPreset': 'BULLET_DISC_CIRCLE_SQUARE'
        }})
    for (r_start, r_end) in contiguous_ranges('file_item'):
        list_requests.append({'createParagraphBullets': {
            'range': {'startIndex': content_start_index + r_start, 'endIndex': content_start_index + r_end},
            'bulletPreset': 'BULLET_DISC_CIRCLE_SQUARE'
        }})
    if list_requests:
        docs_service.documents().batchUpdate(documentId=doc_id, body={'requests': list_requests}).execute()

    # ---------- 4. DIAGRAM IMAGE ----------
    # Note: diagram_url was already validated as a real image when it was generated
    # (in open_source_diagram_generator_tool) -- we deliberately don't re-check it
    # here, since a second round-trip to Kroki's free public server just doubles
    # exposure to transient flakiness for a URL that was already confirmed good.
    if diagram_url:
        doc_now = docs_service.documents().get(documentId=doc_id, fields='body.content').execute()
        end_index = doc_now['body']['content'][-1]['endIndex'] - 1
        try:
            docs_service.documents().batchUpdate(
                documentId=doc_id,
                body={'requests': [{'insertInlineImage': {
                    'location': {'index': end_index},
                    'uri': diagram_url,
                    'objectSize': {'height': {'magnitude': 350, 'unit': 'PT'}, 'width': {'magnitude': 450, 'unit': 'PT'}}
                }}]}
            ).execute()
        except Exception as image_error:
            print(f"⚠️ Diagram image insert failed, Doc created without it: {image_error}")
            sys.stdout.flush()

    # ---------- 5. HEADER LOGO + FOOTER (repeat on every page) ----------
    try:
        header_footer = docs_service.documents().batchUpdate(
            documentId=doc_id,
            body={'requests': [
                {'createHeader': {'type': 'DEFAULT'}},
                {'createFooter': {'type': 'DEFAULT'}}
            ]}
        ).execute()
        replies = header_footer.get('replies', [])
        header_id = replies[0].get('createHeader', {}).get('headerId')
        footer_id = replies[1].get('createFooter', {}).get('footerId')

        logo_url = get_logo_url(credentials, logo_path)
        if logo_url and header_id:
            docs_service.documents().batchUpdate(
                documentId=doc_id,
                body={'requests': [{'insertInlineImage': {
                    'location': {'segmentId': header_id, 'index': 0},
                    'uri': logo_url,
                    'objectSize': {'height': {'magnitude': 24, 'unit': 'PT'}, 'width': {'magnitude': 90, 'unit': 'PT'}}
                }}]}
            ).execute()

        if footer_id:
            footer_text = "Generated by DocAgent"
            docs_service.documents().batchUpdate(
                documentId=doc_id,
                body={'requests': [
                    {'insertText': {'location': {'segmentId': footer_id, 'index': 0}, 'text': footer_text}},
                    {'updateTextStyle': {
                        'range': {'segmentId': footer_id, 'startIndex': 0, 'endIndex': len(footer_text)},
                        'textStyle': {'italic': True, 'fontSize': {'magnitude': 9, 'unit': 'PT'}, 'weightedFontFamily': {'fontFamily': FIGTREE, 'weight': 400}},
                        'fields': 'italic,fontSize,weightedFontFamily'
                    }}
                ]}
            ).execute()
    except Exception as header_error:
        print(f"⚠️ Could not set up header/footer (doc still usable without it): {header_error}")
        sys.stdout.flush()

    # ---------- 6. LINK SHARING ----------
    try:
        drive_service.permissions().create(fileId=doc_id, body={'type': 'anyone', 'role': 'reader'}).execute()
    except Exception as perm_error:
        print(f"⚠️ Could not set link-sharing (doc still accessible to folder members): {perm_error}")
        sys.stdout.flush()

    return f"https://docs.google.com/document/d/{doc_id}/edit"

@tool
def open_source_diagram_generator_tool(mermaid_code: str) -> str:
    """Generates a compressed live Mermaid image URL using Kroki to stay safely under the 2KB Google API limit."""
    try:
        clean_code = sanitize_mermaid(mermaid_code)

        # Compress the text using zlib deflate (Level 9 max compression)
        compressed_data = zlib.compress(clean_code.encode('utf-8'), level=9)

        # Encode using URL-safe Base64 as required by the Kroki specification
        encoded_string = base64.urlsafe_b64encode(compressed_data).decode('utf-8')

        # Route to Kroki's high-performance microservice to return a production-grade PNG
        return f"https://kroki.io/mermaid/png/{encoded_string}"
    except Exception as e:
        print(f"⚠️ Diagram tool compression failed: {e}")
        sys.stdout.flush()
        return ""

# =====================================================================
# 2. PRIVATE REPOSITORY INGESTOR
# =====================================================================
class GitHubRateLimitError(Exception):
    """Raised when GitHub's primary or secondary rate limit is hit, with a clear ETA."""
    pass

class AuthenticatedRepoIngestor:
    def __init__(self, pat_token: str, max_files: int = 400, max_total_chars: int = 600_000):
        self.pat_token = pat_token
        self.IGNORED_DIRS = {
            'node_modules', '.git', '.github', 'venv', 'env', '__pycache__',
            '.pytest_cache', '.egg-info', 'dist', 'build', 'vendor', 'target',
            '.next', '.nuxt', 'coverage', '.venv'
        }
        # Broadened beyond Python-only so this works reasonably well on any
        # public repo's primary language, not just Python/config-heavy ones.
        self.ALLOWED_EXTENSIONS = {
            '.py', '.json', '.yaml', '.yml', '.md', '.txt', '.sh', 'Dockerfile',
            '.js', '.jsx', '.ts', '.tsx', '.go', '.java', '.rb', '.php', '.rs',
            '.c', '.cpp', '.h', '.hpp', '.cs', '.swift', '.kt', '.scala',
            '.sql', '.graphql', '.proto', '.toml', '.cfg', '.ini'
        }
        # Pure noise even if the extension matches -- never useful for architecture understanding
        self.EXCLUDED_FILENAME_SUBSTRINGS = {
            '.min.js', '.min.css', 'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
            'poetry.lock', '.snap', '.generated.', '.pb.go'
        }
        # Entry-point-ish filenames get ingested first, so if the budget runs out,
        # what's cut is the least architecturally important content, not the most.
        self.PRIORITY_FILENAMES = {
            'readme.md', 'main.py', 'app.py', 'index.js', 'index.ts', 'main.go',
            'main.java', 'server.py', 'server.js', 'settings.py', 'config.py',
            'docker-compose.yml', 'dockerfile'
        }
        self.MAX_FILES = max_files
        self.MAX_TOTAL_CHARS = max_total_chars  # conservative budget re: Gemini free-tier TPM
        self.MAX_SINGLE_FILE_CHARS = 40_000      # one huge generated/vendored file can't eat the whole budget
        self.PER_REQUEST_DELAY = 0.05            # small delay between per-file fetches to avoid GitHub's secondary abuse-rate limit
        self.last_repo_meta = None

    def _parse_url(self, url: str) -> dict:
        match = re.search(r"github\.com/([^/]+)/([^/]+)", url)
        if not match: raise ValueError("Invalid URL path format.")
        return {"owner": match.group(1), "repo": match.group(2).replace(".git", "").split("/")[0]}

    def _check_rate_limit(self, response):
        """Raises a clear, user-facing error if GitHub's primary or secondary
        rate limit was hit, with an ETA computed from the response headers."""
        if response.status_code != 403:
            return
        remaining = response.headers.get("X-RateLimit-Remaining")
        if remaining == "0":
            reset_ts = response.headers.get("X-RateLimit-Reset")
            if reset_ts:
                wait_minutes = max(1, int((int(reset_ts) - time.time()) / 60))
                raise GitHubRateLimitError(
                    f"GitHub's hourly API limit was hit. Try again in about {wait_minutes} minute(s)."
                )
            raise GitHubRateLimitError("GitHub's hourly API limit was hit. Try again in a bit.")
        body_text = response.text.lower()
        if "secondary rate limit" in body_text or "abuse" in body_text:
            retry_after = response.headers.get("Retry-After")
            wait_s = int(retry_after) if retry_after else 60
            raise GitHubRateLimitError(
                f"GitHub is temporarily throttling rapid requests. Try again in about {wait_s} seconds."
            )

    def _is_excluded(self, path: str) -> bool:
        lower = path.lower()
        return any(sub in lower for sub in self.EXCLUDED_FILENAME_SUBSTRINGS)

    def compile_buffer(self, url: str) -> str:
        meta = self._parse_url(url)
        headers = {"User-Agent": "DocAgent", "Authorization": f"token {self.pat_token}"}

        with httpx.Client() as client:
            repo_res = client.get(f"https://api.github.com/repos/{meta['owner']}/{meta['repo']}", headers=headers)
            self._check_rate_limit(repo_res)
            if repo_res.status_code != 200:
                raise Exception("Access Denied to Repo (private repo the PAT can't see, or repo doesn't exist).")
            branch = repo_res.json().get("default_branch", "main")
            self.last_repo_meta = {"owner": meta["owner"], "repo": meta["repo"], "branch": branch}

            tree_res = client.get(
                f"https://api.github.com/repos/{meta['owner']}/{meta['repo']}/git/trees/{branch}?recursive=1",
                headers=headers
            )
            self._check_rate_limit(tree_res)
            if tree_res.status_code != 200:
                raise Exception("Failed to read repository file tree.")
            entries = tree_res.json().get("tree", [])

            # Build candidate file list, filtered by extension/exclusions, priority files first
            candidates = []
            for item in entries:
                path = item.get("path", "")
                if item.get("type") != "blob":
                    continue
                if any(d in path.split("/") for d in self.IGNORED_DIRS):
                    continue
                if self._is_excluded(path):
                    continue
                if not any(path.endswith(ext) or path.lower() == 'dockerfile' for ext in self.ALLOWED_EXTENSIONS):
                    continue
                candidates.append(path)

            candidates.sort(key=lambda p: 0 if os.path.basename(p).lower() in self.PRIORITY_FILENAMES else 1)

            buffer = [f"METADATA: Repo: {meta['repo']}\n\n=== SOURCE CODES ===\n"]
            total_chars = 0
            files_included = 0
            files_truncated_for_budget = 0

            for path in candidates:
                if files_included >= self.MAX_FILES or total_chars >= self.MAX_TOTAL_CHARS:
                    files_truncated_for_budget += 1
                    continue

                res = client.get(f"https://raw.githubusercontent.com/{meta['owner']}/{meta['repo']}/{branch}/{path}", headers=headers)
                time.sleep(self.PER_REQUEST_DELAY)
                self._check_rate_limit(res)
                if res.status_code != 200:
                    continue

                try:
                    text = res.text
                except Exception:
                    continue  # binary or undecodable content -- skip rather than crash the whole run

                if not text.strip():
                    continue
                if len(text) > self.MAX_SINGLE_FILE_CHARS:
                    text = text[:self.MAX_SINGLE_FILE_CHARS] + "\n... [truncated: file too large] ..."

                buffer.append(f"--- START: {path} ---\n{text}\n--- END ---\n")
                total_chars += len(text)
                files_included += 1

            if files_truncated_for_budget > 0:
                buffer.append(
                    f"\n[NOTE: repository is large -- {files_truncated_for_budget} additional file(s) "
                    f"were omitted to stay within processing limits. Architecture summary reflects the "
                    f"{files_included} most relevant files.]\n"
                )

            return "\n".join(buffer)

# =====================================================================
# 3. ORCHESTRATOR
# =====================================================================
class ProductionDocOrchestrator:
    def __init__(self):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_KEY"))
        self.ingestor = AuthenticatedRepoIngestor(pat_token=os.environ.get("GITHUB_PAT"))

    def _generate_json(self, prompt: str, system_instruction: str, schema, max_output_tokens: int = 32768):
        """Shared retry/repair logic for any Gemini structured-JSON call: handles
        truncated JSON (retries) and transient 503 overload (backoff retries)."""
        parsed_output = None
        last_error = None

        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        system_instruction=system_instruction,
                        max_output_tokens=max_output_tokens,
                        temperature=0.0
                    ),
                )

                raw_text = response.text.strip()
                if raw_text.startswith("```"):
                    raw_text = re.sub(r'^```(?:json)?\n', '', raw_text, flags=re.IGNORECASE)
                    raw_text = re.sub(r'\n```$', '', raw_text)
                    raw_text = raw_text.strip()

                parsed_output = json.loads(raw_text, strict=False)
                break

            except json.JSONDecodeError as e:
                last_error = e
                print(f"⚠️ AI generated truncated JSON (attempt {attempt + 1}/3), retrying...")
                sys.stdout.flush()
                continue

            except Exception as e:
                last_error = e
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    wait_seconds = 2 ** attempt
                    print(f"⏳ Gemini overloaded (attempt {attempt + 1}/3), retrying in {wait_seconds}s...")
                    sys.stdout.flush()
                    time.sleep(wait_seconds)
                    continue
                raise

        if parsed_output is None:
            raise Exception(f"Gemini failed to produce valid JSON after 3 attempts: {last_error}")

        return parsed_output

    def run_pipeline(self, target_url: str):
        codebase_context = self.ingestor.compile_buffer(target_url)

        chat_system_instruction = """
        You are a rigid production systems architect. Generate a unified, strictly factual infrastructure and application architecture diagram based on the parsed codebase context.
        Absolutely no creative embellishments or non-existent components are allowed.

        CRITICAL TOPOLOGY RULES FOR diagram_code (Must be completely valid inside a JSON string):
        1. Layout: Must start with `flowchart LR`.

        2. Static Hosting Infrastructure Block: You MUST include this exact subgraph structure at the beginning of the diagram.
           CRITICAL: DO NOT USE QUOTES OF ANY KIND. Do not use punctuation or parentheses inside text labels.
           subgraph Hosting_Infrastructure [GCP Production Infrastructure Cloud Run]
               ChatUser[Google Chat Interface] -->|1 Async POST| WebhookEP[Cloud Run Webhook Endpoint]
               WebhookEP -->|2 Check Idempotency| FirestoreDB[Firestore Cache]
               WebhookEP -->|3 Offload Job| CloudTasks[Cloud Tasks Queue]
               CloudTasks -->|4 Secure Trigger| WorkerEngine[Cloud Run Worker Engine]
               WorkerEngine -->|5 Read Secrets| SecretManager[Google Secret Manager]
           end

        3. Dynamic Application Blocks: Extract the functional components from the scanned codebase context and group them into these three specific subgraphs (NO QUOTES):
           - subgraph Ingestion_Tier [Application Ingestion Tier]
           - subgraph Processing_Tier [Application Processing Tier]
           - subgraph Delivery_Tier [Application Delivery Tier]

        4. Loop-Closure Connections: You MUST explicitly draw these structural edge routing links to bind the infrastructure to the application:
           - Draw a connection from WorkerEngine to the initial code entry node inside Ingestion_Tier.
           - Draw a connection from the final response node of Delivery_Tier back to ChatUser.
           - Draw a connection from the final processing node of Delivery_Tier to a standalone node named GDoc[Google Workspace Doc Asset].

        5. Syntax & Compliance Restraints:
           - Node IDs MUST start with a letter.
           - Every single component declaration, statement, and connection step MUST be written on its own individual line.
           - DO NOT USE QUOTES (single or double) anywhere in the diagram text.
           - DO NOT USE parentheses (), braces {}, or special punctuation inside label brackets. Use clean alphanumeric characters and spaces only.
           - Max 10 total logical nodes across the three application tiers combined to ensure scannability.
        """

        parsed_output = self._generate_json(
            prompt=f"Populate the JSON schema from this codebase:\n\n{codebase_context}",
            system_instruction=chat_system_instruction,
            schema=SpecSchema,
            max_output_tokens=32768
        )

        doc_text = parsed_output.get("document_content", [])
        diagram_link = open_source_diagram_generator_tool.invoke({"mermaid_code": parsed_output.get("diagram_code", "")})

        return {
            "document": doc_text,
            "image_url": diagram_link,
            "codebase_context": codebase_context,
            "repo_meta": self.ingestor.last_repo_meta
        }

    def generate_deep_doc_content(self, codebase_context: str) -> dict:
        """Separate, dedicated Gemini call for the full-length Doc -- distinct from the
        short Chat-card bullets. Asks for the same depth of output a manual agentic
        code-exploration session would produce: an overview, an ordered execution flow,
        thematic sections, and a file-by-file breakdown grouped by directory."""
        doc_system_instruction = """
        You are a senior software architect producing a formal, deeply detailed architecture
        document for engineers who were not involved in building this system -- the kind of
        document a thorough manual code review would produce, not a shallow surface summary.
        Write in full, clear prose. Reference actual filenames, function names, models, and
        technologies found in the codebase context throughout -- never invent components that
        aren't present in the code.

        Produce ALL of the following:

        1. project_name: the system's actual name, inferred from the repo.

        2. overview: 4-6 sentences on what this codebase is, its primary purpose, and its core
           technical approach.

        3. execution_flow: 4-8 ordered steps describing what actually happens end-to-end, in
           the real order it happens, e.g. "Step 1: X uploads via Y, which is handled by Z.py"
           -- specific, not generic.

        4. sections: 6-10 thematic sections (adapt names to what's actually present -- e.g. for
           a data pipeline: Data Sources & Ingestion, Processing, Storage, API Layer, Caching,
           Orchestration; for an ML/vision system: Feature Extraction, Model Inference, Synthesis
           Layer). Each section: a clear heading and 2-4 full sentences of real explanation.

        5. file_breakdown: group EVERY meaningfully analyzed file by its directory/module. For
           each file, 1-2 sentences on exactly what it does and why it matters -- not just a
           restated filename. This should be as close to exhaustive as the codebase context
           allows -- the goal is a genuine file-by-file map a new engineer could use to
           navigate the repo, not a token summary.

        6. use_cases: 3-6 concrete real-world use cases this system enables for the people who'd
           actually use it.
        """

        parsed_output = self._generate_json(
            prompt=f"Write the full architecture document for this codebase:\n\n{codebase_context}",
            system_instruction=doc_system_instruction,
            schema=DeepDocSchema,
            max_output_tokens=32768
        )

        return parsed_output

# =====================================================================
# 4. BACKGROUND PROCESSING FUNCTION
# =====================================================================
def process_and_reply(target_repo: str, space_name: str, credentials_path: str, requester_name: str = "DocAgent"):
    """Processes the request in the background and posts back via the Google Chat API."""
    try:
        print(f"🔄 Thread started for {target_repo}...")
        sys.stdout.flush()

        # FIXED: plain scope URL, no Markdown link wrapping
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=CHAT_SCOPES
        )
        chat_service = build('chat', 'v1', credentials=credentials)

        repo_name = target_repo.split('/')[-1].replace('.git', '')

        # Send initial confirmation message inside the background worker
        chat_service.spaces().messages().create(
            parent=space_name,
            body={"text": f"⏳ I am compiling the codebase for `{repo_name}`. Generating blueprint..."}
        ).execute()
        print("✅ Loading status message pushed to chat channel.")
        sys.stdout.flush()

        orchestrator = ProductionDocOrchestrator()
        results = orchestrator.run_pipeline(target_repo)

        bullet_points = "<br>".join([f"• {point}" for point in results['document']])

        # Try to create a formatted Google Doc using user context credentials.
        # The Doc gets its own dedicated, deeper generation pass -- not the same
        # short bullets used in the Chat card -- so it reads like a real document.
        doc_url = None
        try:
            user_credentials = get_user_credentials()
            deep_content = orchestrator.generate_deep_doc_content(results['codebase_context'])

            github_base_url = None
            repo_meta = results.get('repo_meta')
            if repo_meta:
                github_base_url = f"https://github.com/{repo_meta['owner']}/{repo_meta['repo']}/blob/{repo_meta['branch']}/"

            doc_url = create_google_doc(
                user_credentials,
                title=deep_content.get('project_name') or repo_name,
                subtitle="Architecture Blueprint",
                author_name=requester_name,
                sections=deep_content.get('sections', []),
                overview=deep_content.get('overview'),
                execution_flow=deep_content.get('execution_flow', []),
                file_breakdown=deep_content.get('file_breakdown', []),
                use_cases=deep_content.get('use_cases', []),
                github_base_url=github_base_url,
                diagram_url=results['image_url']
            )
            print(f"📄 Google Doc successfully created: {doc_url}")
            sys.stdout.flush()
        except Exception as doc_error:
            print(f"⚠️ Doc creation failed (continuing without it): {doc_error}")
            sys.stdout.flush()

        buttons = []
        if doc_url:
            buttons.append({"text": "📄 Open Full Report (Doc)", "onClick": {"openLink": {"url": doc_url}}})
        if results['image_url']:
            buttons.append({"text": "🔍 View Full-Size Diagram", "onClick": {"openLink": {"url": results['image_url']}}})

        card_sections = [
            {"widgets": [{"textParagraph": {"text": f"<b>System Specifications:</b><br>{bullet_points}"}}]}
        ]
        if results['image_url']:
            card_sections.append({"widgets": [{"image": {"imageUrl": results['image_url'], "altText": "Architecture Map"}}]})
        if buttons:
            card_sections.append({"widgets": [{"buttonList": {"buttons": buttons}}]})

        card_payload = {
            "cardsV2": [{
                "cardId": "docagent_architecture_card",
                "card": {
                    "header": {
                        "title": "Architecture Blueprint",
                        "subtitle": f"Target: {repo_name}"
                    },
                    "sections": card_sections
                }
            }]
        }

        chat_service.spaces().messages().create(
            parent=space_name,
            body=card_payload
        ).execute()
        print("✅ Architecture Map successfully injected into chat!")
        sys.stdout.flush()

    except Exception as e:
        print(f"❌ Background error: {e}")
        sys.stdout.flush()
        try:
            # FIXED: plain scope URL, no Markdown link wrapping
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path, scopes=CHAT_SCOPES
            )
            chat_service = build('chat', 'v1', credentials=credentials)
            chat_service.spaces().messages().create(
                parent=space_name,
                body={"text": f"⚠️ Couldn't finish mapping that repo: {str(e)[:200]}"}
            ).execute()
        except Exception as inner_e:
            print(f"❌ Failed to even send the error message: {inner_e}")
            sys.stdout.flush()

# =====================================================================
# 5. GOOGLE CHAT FLASK ROUTE
# =====================================================================
@app.route('/', methods=['POST'])
def google_chat_webhook():
    event = request.get_json()
    if not event:
        print("⚠️ Received an empty or malformed POST request.")
        sys.stdout.flush()
        return jsonify({})

    # Clean Logging: Drops the heavy authorization payload to keep the terminal readable
    clean_event = {k: v for k, v in event.items() if k != 'authorizationEventObject'}
    print(f"📬 Incoming Event Payload: {json.dumps(clean_event)}")
    sys.stdout.flush()

    # Safely extracting variables from the nested modern Google schema layout
    chat_obj = event.get('chat', {})
    message_payload = chat_obj.get('messagePayload', {})
    message_obj = message_payload.get('message', {})

    user_text = message_obj.get('text', '')
    space_name = message_payload.get('space', {}).get('name', '')
    sender_name = message_obj.get('sender', {}).get('displayName', 'DocAgent')

    # Upgraded Regex: Clean extraction that discards hidden markdown angle brackets if present
    url_match = re.search(r'https://github\.com/[a-zA-Z0-9_\.\-]+/[a-zA-Z0-9_\.\-]+', user_text)

    if url_match:
        target_repo = url_match.group(0).split('>')[0].strip()
        print(f"🎯 Target Repository Extracted: {target_repo}")
        sys.stdout.flush()

        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

        if not credentials_path:
            print("⚠️ ERROR: Missing GOOGLE_APPLICATION_CREDENTIALS in .env!")
            sys.stdout.flush()
            return jsonify({"text": "❌ Internal Bot Error: Missing server credentials key."})

        # Launch non-blocking background worker thread
        thread = threading.Thread(target=process_and_reply, args=(target_repo, space_name, credentials_path, sender_name))
        thread.start()

        # Returning a clean 200 OK empty object eliminates the "not responding" ghost banner
        return jsonify({})

    print(f"❓ Regex missed. Text received: '{user_text}'")
    sys.stdout.flush()
    return jsonify({"text": "Hi! Paste a valid GitHub repository URL, and I'll analyze and map the architecture for you."})

# =====================================================================
# 6. ENGINE IGNITION SWITCH
# =====================================================================
if __name__ == '__main__':
    app.run(port=8080, debug=False, threaded=True)