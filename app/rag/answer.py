from pathlib import Path
from typing import List, Any

from app.rag.llm import _safe_llm_call
from app.utils.misc import looks_like_gibberish


def generate_answer_with_llm(llm_obj, question: str, hits: List[Any]) -> str:
    q_lower = question.lower()
    wants_summary = any(
        k in q_lower
        for k in ["summary", "explain", "describe", "details", "what is", "who", "why", "how"]
    )
    wants_image = any(
        k in q_lower
        for k in ["image", "photo", "picture", "show", "display"]
    )

    parts = []
    file_types = set()

    for d, s in hits[:6]:
        meta = d.metadata or {}
        path = meta.get("path", "<no-path>")
        ftype = meta.get("type", "").lower()
        file_types.add(ftype)

        parts.append(
            {
                "path": path,
                "type": ftype,
                "summary": meta.get("summary", ""),
                "excerpt": (d.page_content or "").strip(),
                "score": s,
            }
        )

    #Image Handler
    if wants_image or (
        len(file_types) == 1 and list(file_types)[0] in {"jpg", "jpeg", "png"}
    ):
        response_lines = []
        for p in parts[:6]:
            if p["type"] in {"jpg", "jpeg", "png"}:
                name = Path(p["path"]).name
                desc = p["summary"].capitalize() if p["summary"] else "An image file."
                response_lines.append(f"🖼️ {name}: {desc}")
        return "\n".join(response_lines) + "\n---"

    #Context Builder
    context_blocks = []
    for p in parts[:6]:
        ftype = p["type"]
        name = Path(p["path"]).name
        summary = p["summary"][:400]
        excerpt = p["excerpt"][:500]

        if ftype in {"pdf", "docx", "txt", "pptx", "csv"}:
            block = (
                f"📄 File: {name} ({ftype})\n"
                f"Summary: {summary}\n"
                f"Excerpt: {excerpt}\n---"
            )
        else:
            block = f"📁 File: {name}\nInfo: {summary}\n---"

        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    #System Prompt for LLM when Summarizer is Called
    if wants_summary:
        role_prompt = (
            "You are a concise assistant. The user is asking for a summary or explanation.\n"
            "Use the provided file context to summarize key details naturally.\n"
            "Do not include raw file data. Be clear and short (3-5 sentences).\n"
            "Finish with 'Sources:' and include file paths.\n\n"
        )
    else:
        role_prompt = (
            "You are a helpful assistant that answers questions based on the user's local files.\n"
            "Respond naturally and briefly. Mention relevant file types or names if useful.\n"
            "Avoid robotic formatting. End with 'Sources:' listing the files you referenced.\n\n"
        )

    prompt = (
        f"{role_prompt}"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer naturally:"
    )

    try:
        resp_text = _safe_llm_call(llm_obj, prompt).strip()
        return resp_text
    except Exception:
        lines = [
            f"• {p['path']} — {p['summary'] or p['excerpt'][:120]}"
            for p in parts[:6]
        ]
        return "Here are the most relevant files:\n" + "\n".join(lines)
