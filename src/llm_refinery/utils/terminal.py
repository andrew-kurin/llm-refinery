from __future__ import annotations

import unicodedata


def sanitize_terminal_text(value: str) -> str:
    """Strip terminal control strings, escape sequences, and Unicode controls."""
    result: list[str] = []
    index = 0
    while index < len(value):
        codepoint = ord(value[index])
        if codepoint == 0x1B:
            index = _skip_escape_sequence(value, index + 1)
            continue
        if codepoint in {0x90, 0x98, 0x9D, 0x9E, 0x9F}:
            index = _skip_control_string(value, index + 1, osc=codepoint == 0x9D)
            continue
        if codepoint == 0x9B:
            index = _skip_csi(value, index + 1)
            continue
        if codepoint < 0x20:
            if value[index] in "\t\n\r":
                result.append(" ")
            index += 1
            continue
        category = unicodedata.category(value[index])
        if category in {"Zl", "Zp"}:
            result.append(" ")
            index += 1
            continue
        if 0x7F <= codepoint <= 0x9F or category in {"Cf", "Cs"}:
            index += 1
            continue
        result.append(value[index])
        index += 1
    return "".join(result)


def _skip_escape_sequence(value: str, index: int) -> int:
    if index >= len(value):
        return index
    introducer = value[index]
    if introducer == "[":
        return _skip_csi(value, index + 1)
    if introducer == "]":
        return _skip_control_string(value, index + 1, osc=True)
    if introducer in {"P", "X", "^", "_"}:
        return _skip_control_string(value, index + 1, osc=False)

    # ANSI two-byte and intermediate escape sequences end in 0x30-0x7e.
    while index < len(value) and 0x20 <= ord(value[index]) <= 0x2F:
        index += 1
    if index < len(value) and 0x30 <= ord(value[index]) <= 0x7E:
        index += 1
    return index


def _skip_csi(value: str, index: int) -> int:
    while index < len(value):
        codepoint = ord(value[index])
        index += 1
        if 0x40 <= codepoint <= 0x7E:
            break
    return index


def _skip_control_string(value: str, index: int, *, osc: bool) -> int:
    while index < len(value):
        codepoint = ord(value[index])
        if (osc and codepoint == 0x07) or codepoint == 0x9C:
            return index + 1
        if codepoint == 0x1B and index + 1 < len(value) and value[index + 1] == "\\":
            return index + 2
        index += 1
    return index


__all__ = ["sanitize_terminal_text"]
