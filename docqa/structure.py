"""Restore explicit Chinese outline hierarchy when layout models flatten heading levels."""

import re

OUTLINE_VERSION = "explicit-chinese-outline-v1"


def restore_section_paths(elements):
    stacks = {}
    restored = []
    for element in elements:
        stack = stacks.setdefault(element.document_id, [])
        path = element.title_path
        text = element.text.strip()
        # Only use items already labelled as headings; never infer a heading from body numbers.
        if path and text == path[-1].strip():
            compact = re.sub(r"\s+", "", text)
            if re.match(r"^[一二三四五六七八九十百]+[、．.]", compact):
                level = 1
            elif re.match(r"^[（(][一二三四五六七八九十百]+[）)]", compact):
                level = 2
            else:
                # Preserve the parser's hierarchy for unrecognised numbering conventions.
                stack[:] = path
                restored.append(element)
                continue
            stack[:] = stack[: level - 1] + [text]
        if path and stack and path[-1].strip() == stack[-1].strip():
            element = element.model_copy(update={"title_path": list(stack)})
        restored.append(element)
    return restored
