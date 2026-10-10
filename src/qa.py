"""qa.py - retrieve (CV first), ask the LLM, return answer + sources + chunks."""
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from src.vectordb_creation import search

load_dotenv()

MODELS = [m.strip() for m in os.getenv("OPENROUTER_MODELS", "").split(",") if m.strip()]
TOP_K = 6        # total passages sent to the LLM
CV_SLOTS = 3     # of those, how many are reserved for the best CV passages
CV_HINT = "cv"   # a PDF whose file name contains this is treated as the CV

SYSTEM_PROMPT = (
    "You answer questions about one person's professional background.\n"
    "Use ONLY the numbered context passages provided. "
    "If the answer is not in them, say you don't have that information.\n"
    "Passages whose source is the CV are the most authoritative: "
    "if sources disagree, trust the CV.\n"
    "Cite the passages you used with their numbers, like [1] or [2][3].\n"
    "Be concise and factual. Do not invent details."
)


def _client():
    key = os.getenv("OPENROUTER_API_KEY")
    if not key or not MODELS:
        raise RuntimeError("Set OPENROUTER_API_KEY and OPENROUTER_MODELS in .env")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def is_cv(meta):
    return meta.get("type") == "pdf" and CV_HINT in meta.get("source", "").lower()


def _label(meta):
    """Turn chunk metadata into a human-readable citation."""
    if meta.get("type") == "pdf":
        return f"{meta['source']}, p.{meta['page']}"
    return meta["source"]  # GitHub: the repo URL


def retrieve(question, k=TOP_K, cv_slots=CV_SLOTS):
    """CV first: best CV passages get reserved slots, other sources fill the rest."""
    cv_hits = search(question, k=min(cv_slots, k), where=is_cv)
    seen = {doc.metadata["chunk_id"] for doc, _ in cv_hits}
    merged = list(cv_hits)

    # unrestricted search, asking for extra so duplicates can be skipped
    for doc, score in search(question, k=k + cv_slots):
        if len(merged) >= k:
            break
        if doc.metadata["chunk_id"] in seen:
            continue
        seen.add(doc.metadata["chunk_id"])
        merged.append((doc, score))
    return merged


def _build_context(results):
    """Number each retrieved chunk so the LLM can cite it as [1], [2], ..."""
    blocks = []
    for n, (doc, _score) in enumerate(results, start=1):
        blocks.append(f"[{n}] (source: {_label(doc.metadata)})\n{doc.page_content}")
    return "\n\n".join(blocks)


def answer(question, k=TOP_K):
    """Main function. Returns {"answer", "sources", "chunks"}."""
    results = retrieve(question, k=k)                     # 1. retrieve (CV first)
    context = _build_context(results)                     # 2. build prompt context

    resp = _client().chat.completions.create(             # 3. ask the LLM
        model=MODELS[0],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.2,
        extra_body={"models": MODELS} if len(MODELS) > 1 else None,  # fallbacks
    )
    text = resp.choices[0].message.content.strip()

    # 4. keep only the sources the LLM actually cited (fall back to all of them)
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    picked = [i for i in range(1, len(results) + 1) if i in cited] or range(1, len(results) + 1)
    sources = []
    for i in picked:
        label = _label(results[i - 1][0].metadata)
        if label not in sources:
            sources.append(label)

    chunks = [
        {"label": _label(d.metadata), "score": s, "text": d.page_content}
        for d, s in results
    ]
    return {"answer": text, "sources": sources, "chunks": chunks}


if __name__ == "__main__":
    while True:
        q = input("\nAsk about Ammar (or 'quit'): ").strip()
        if q.lower() in {"quit", "exit", ""}:
            break
        out = answer(q)
        print("\n" + out["answer"])
        print("\nSources:")
        for s in out["sources"]:
            print(" -", s)