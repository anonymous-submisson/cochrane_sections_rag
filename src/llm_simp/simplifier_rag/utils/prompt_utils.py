from llm_simp.simplifier_rag.prompts import (
    SPLIT_EXAMPLE_INTRO,
    SYSTEM_PROMPT_RAG_PLS_HEADER,
    WHOLE_EXAMPLE_INTRO,
)


def _render_bulleted(items, empty_marker):
    if not items:
        return empty_marker
    return "\n".join(f"- {s}" for s in items)


def format_example_block_split(section_name, examples):
    lines = [SPLIT_EXAMPLE_INTRO.format(k=len(examples), section_name=section_name), ""]
    for i, ex in enumerate(examples):
        ex_section = ex.get("section_name") or section_name
        lines.append(f"=== Example {i+1} ===")
        lines.append(f"Abstract {ex_section}:")
        lines.append(ex["complex"] if ex["complex"] else "(empty)")
        lines.append("")
        rewrites = ex.get("pls_rewrites") or []
        inserts = ex.get("pls_inserts") or []
        lines.append(f"PLS rewrites of kept abstract sentences ({ex_section}):")
        lines.append(
            _render_bulleted(
                rewrites,
                "(none — every kept sentence was dropped or this section was empty)",
            )
        )
        lines.append("")
        lines.append(
            f"PLS inserts ({ex_section}, added content with no counterpart in the abstract):"
        )
        lines.append(
            _render_bulleted(inserts, "(none — this example added no new content)")
        )
        lines.append("")
    return "\n".join(lines).rstrip()


def format_example_block_whole(section_name, examples):
    lines = [WHOLE_EXAMPLE_INTRO.format(k=len(examples), section_name=section_name), ""]
    for i, ex in enumerate(examples):
        ex_section = ex.get("section_name") or section_name
        lines.append(f"=== Example {i+1} ===")
        lines.append(f"Abstract {ex_section}:")
        lines.append(ex["complex"] if ex["complex"] else "(empty)")
        lines.append("")
        lines.append(f"PLS {ex_section}:")
        lines.append(ex["pls"] if ex.get("pls") else "(empty)")
        lines.append("")
    return "\n".join(lines).rstrip()


def build_system_prompt(section_name, examples, example_format="split"):
    parts = [SYSTEM_PROMPT_RAG_PLS_HEADER]
    if examples:
        if example_format == "whole":
            parts.append(format_example_block_whole(section_name, examples))
        else:
            parts.append(format_example_block_split(section_name, examples))
    return "\n\n".join(parts)


def build_user_prompt(section_name, section_sents, current_idx):
    lines = [f"Section: {section_name}", ""]
    lines.append("Section text:")
    for i, s in enumerate(section_sents):
        lines.append(f"{i+1}. {s}")
    lines.append("")
    lines.append("Current sentence:")
    lines.append(f"{current_idx+1}. {section_sents[current_idx]}")
    lines.append(">>>")
    return "\n".join(lines)
