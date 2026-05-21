def build_user_prompt(section_sents, current_idx):
    lines = ["Section text:"]
    for i, s in enumerate(section_sents):
        lines.append(f"{i+1}. {s}")

    lines.append("")
    lines.append("Current sentence:")
    lines.append(f"{current_idx+1}. {section_sents[current_idx]}")
    lines.append(">>>")

    return "\n".join(lines)
