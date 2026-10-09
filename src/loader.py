"""loader.py - turn raw sources into LangChain Documents that carry source info."""
import os
from pathlib import Path

import pymupdf  # PyMuPDF
import requests
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

RAW_DIR = Path("data/raw")
GITHUB_API = "https://api.github.com"


# ---------- PDFs (CV + LinkedIn) ----------
def load_pdfs(raw_dir=RAW_DIR):
    docs = []
    for pdf in sorted(Path(raw_dir).glob("*.pdf")):
        with pymupdf.open(pdf) as doc:
            for i, page in enumerate(doc, start=1):  # page numbers start at 1
                text = page.get_text("text", sort=True).strip()
                if not text:
                    continue
                docs.append(Document(
                    page_content=text,
                    metadata={"source": pdf.name, "page": i, "type": "pdf"},
                ))
    return docs


# ---------- GitHub (public repos + READMEs) ----------
def _headers(raw=False):
    accept = "application/vnd.github.raw+json" if raw else "application/vnd.github+json"
    headers = {"Accept": accept}
    token = os.getenv("GITHUB_TOKEN")  # optional: raises the rate limit
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def load_github(username, cache_dir=RAW_DIR / "github", include_forks=False):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    docs, page = [], 1

    while True:
        r = requests.get(
            f"{GITHUB_API}/users/{username}/repos",
            params={"per_page": 100, "page": page, "type": "owner"},
            headers=_headers(),
            timeout=15,
        )
        r.raise_for_status()
        repos = r.json()
        if not repos:
            break

        for repo in repos:
            if repo["fork"] and not include_forks:
                continue

            rm = requests.get(
                f"{GITHUB_API}/repos/{username}/{repo['name']}/readme",
                headers=_headers(raw=True),
                timeout=15,
            )
            readme = rm.text if rm.status_code == 200 else ""

            text = (
                f"GitHub repo: {repo['name']}\n"
                f"URL: {repo['html_url']}\n"
                f"Description: {repo['description'] or 'N/A'}\n"
                f"Language: {repo['language'] or 'N/A'}\n"
                f"Topics: {', '.join(repo.get('topics', [])) or 'N/A'}\n\n"
                f"{readme}"
            )

            # keep a readable copy so you can inspect what gets indexed
            (cache_dir / f"{repo['name']}.md").write_text(text, encoding="utf-8")

            docs.append(Document(
                page_content=text,
                metadata={"source": repo["html_url"], "type": "github", "repo": repo["name"]},
            ))
        page += 1

    return docs


# ---------- Everything together ----------
def load_all(github_username=None):
    docs = load_pdfs()
    username = github_username or os.getenv("GITHUB_USERNAME")
    if username:
        docs += load_github(username)
    return docs


if __name__ == "__main__":
    all_docs = load_all()
    print(f"Loaded {len(all_docs)} documents")
    for d in all_docs[:3]:
        print(d.metadata, "->", d.page_content[:200].replace("\n", " "))