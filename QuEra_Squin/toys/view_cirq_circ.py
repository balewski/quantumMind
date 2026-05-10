#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

"""Print a Cirq text circuit with compact numeric gate labels."""

import argparse
import re
import shutil
from pathlib import Path


class ViewNoisyCirqCirc:
    """Format a saved Cirq text circuit for compact terminal viewing."""

    GATE_BASE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    QUBIT_LABEL_RE = re.compile(r"^(\s*\d+:\s*)")
    IDLE_CHARS = {" ", "-", "─"}

    def __init__(self, cirq_text: str):
        """Store input text and precompute numeric gate aliases."""
        self.cirq_text = cirq_text
        self.aliases = self._find_numeric_gate_aliases(cirq_text)

    def emit_circ(self, lineLen: int, wireLen: int) -> str:
        """Return transformed circuit text split to lineLen columns."""
        circuit_text = self._format_cirq_text(self.cirq_text, self.aliases)
        circuit_text = self._compact_idle_wires(circuit_text, wireLen)
        return self._circuit_sections_text(circuit_text, lineLen)

    def gate_aliases(self) -> str:
        """Return alphabetized gate-alias definitions or an empty string."""
        return self._format_alias_definitions(self.aliases) if self.aliases else ""

    @staticmethod
    def _replace_token_keep_width(line: str, old: str, new: str) -> str:
        """Replace a gate token and pad with wire dashes to keep columns aligned."""
        assert len(new) <= len(old)
        return line.replace(old, new + "─" * (len(old) - len(new)))

    @staticmethod
    def _alias_letter(index: int) -> str:
        """Return spreadsheet-style lowercase alias letters: a, b, ..., aa, ab."""
        letters = []
        while True:
            index, rem = divmod(index, 26)
            letters.append(chr(ord("a") + rem))
            if index == 0:
                return "".join(reversed(letters))
            index -= 1

    @classmethod
    def _gate_base_name(cls, token: str) -> str:
        """Return the leading alphabetic gate name before numeric parameters."""
        return cls.GATE_BASE_RE.match(token).group(0)

    @classmethod
    def _find_numeric_gate_tokens(cls, text: str) -> list:
        """Return numeric gate tokens, including full A(...) style labels."""
        tokens = []
        i = 0
        while i < len(text):
            match = cls.GATE_BASE_RE.match(text, i)
            if match is None:
                i += 1
                continue

            start = i
            stop = match.end()
            if stop < len(text) and text[stop] == "(":
                depth = 1
                stop += 1
                while stop < len(text) and depth > 0:
                    if text[stop] == "(":
                        depth += 1
                    elif text[stop] == ")":
                        depth -= 1
                    stop += 1
            elif stop < len(text) and text[stop] == "^":
                stop += 1
                if stop < len(text) and text[stop] in "+-":
                    stop += 1
                while stop < len(text) and (text[stop].isdigit() or text[stop] == "."):
                    stop += 1

            token = text[start:stop]
            if any(char.isdigit() for char in token):
                tokens.append(token)
            i = stop
        return tokens

    @classmethod
    def _find_numeric_gate_aliases(cls, text: str) -> dict:
        """Map numeric gate tokens to compact aliases in first-seen order."""
        aliases = {}
        base_counts = {}
        for token in cls._find_numeric_gate_tokens(text):
            if token not in aliases:
                base = cls._gate_base_name(token)
                index = base_counts.get(base, 0)
                aliases[token] = base + cls._alias_letter(index)
                base_counts[base] = index + 1
        return aliases

    @classmethod
    def _format_cirq_text(cls, text: str, aliases: dict) -> str:
        """Apply compact numeric gate aliases without changing circuit width."""
        for old, new in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
            text = "\n".join(
                cls._replace_token_keep_width(line, old, new)
                for line in text.splitlines()
            )
        return text

    @classmethod
    def _compact_idle_wires(cls, text: str, wire_len: int) -> str:
        """Shrink global idle-wire column runs to at most wire_len columns."""
        if wire_len <= 0:
            return text

        lines = text.splitlines()
        max_len = max((len(line) for line in lines), default=0)
        padded = [line.ljust(max_len) for line in lines]
        keep_cols = []
        col = 0
        while col < max_len:
            idle = all(row[col] in cls.IDLE_CHARS for row in padded)
            if not idle:
                keep_cols.append(col)
                col += 1
                continue

            start = col
            while col < max_len and all(row[col] in cls.IDLE_CHARS for row in padded):
                col += 1
            keep_cols.extend(range(start, min(col, start + wire_len)))

        return "\n".join("".join(row[col] for col in keep_cols).rstrip() for row in padded)

    @staticmethod
    def _format_alias_definitions(aliases: dict) -> str:
        """Return compact gate alias definitions."""
        lines = ["Gates aliases:"]
        for old, new in sorted(aliases.items(), key=lambda item: item[1]):
            lines.append(f"{new} = {old}")
        return "\n".join(lines)

    @classmethod
    def _circuit_sections_text(cls, text: str, width: int) -> str:
        """Return a fixed-width circuit split into horizontal sections."""
        if width <= 0:
            return text

        lines = text.splitlines()
        out_lines = []
        label_width = 0
        for line in lines:
            match = cls.QUBIT_LABEL_RE.match(line)
            if match:
                label_width = max(label_width, len(match.group(1)))
        if label_width == 0:
            max_len = max((len(line) for line in lines), default=0)
            for start in range(0, max_len, width):
                if start > 0:
                    out_lines.append("")
                for line in lines:
                    out_lines.append(line[start:start + width].rstrip())
            return "\n".join(out_lines)

        rows = []
        repeat_width = label_width + 1
        for line in lines:
            match = cls.QUBIT_LABEL_RE.match(line)
            if match:
                first_prefix = match.group(1)
                repeat_prefix = first_prefix.replace(":", ">>", 1)
                body = line[len(first_prefix):]
                is_qubit_row = True
            else:
                first_prefix = line[:label_width].ljust(label_width)
                repeat_prefix = " " * repeat_width
                body = line[label_width:]
                is_qubit_row = False
            rows.append((first_prefix, repeat_prefix, body, is_qubit_row))

        body_width = max(1, width - label_width)
        max_len = max((len(body) for _, _, body, _ in rows), default=0)
        for start in range(0, max_len, body_width):
            chunks = [body[start:start + body_width] for _, _, body, _ in rows]
            trailing_idle_section = (
                start > 0
                and start + body_width >= max_len
                and all(all(char in cls.IDLE_CHARS for char in chunk) for chunk in chunks)
            )
            if trailing_idle_section:
                continue
            if start > 0:
                out_lines.append("")
            for (first_prefix, repeat_prefix, _, is_qubit_row), chunk in zip(rows, chunks):
                prefix = first_prefix if start == 0 else repeat_prefix
                suffix = ">>" if is_qubit_row and start + body_width < max_len else ""
                out_lines.append((prefix + chunk).rstrip() + suffix)
        return "\n".join(out_lines)


def main() -> None:
    """Read a saved Cirq text circuit and print the reformatted circuit."""
    parser = argparse.ArgumentParser(description=__doc__)
    prs = parser.add_argument
    prs("--name", required=True, help="path to the saved Cirq text circuit")
    prs("--lineLen", type=int, default=None,
        help="horizontal circuit section width; default is terminal width - 2; use 0 for no splitting")
    prs("--wireLen", type=int, default=2,
        help="maximum length for idle wire stretches between active columns")
    prs("-v", "--verb", type=int, default=2,
        help="verbosity: 1 circuit only, 2 print gate aliases above circuit")
    args = parser.parse_args()
    if args.lineLen is None:
        args.lineLen = max(10, shutil.get_terminal_size().columns - 2)

    for arg in vars(args):
        print("myArg:", arg, getattr(args, arg))

    cirq_text = Path(args.name).read_text()
    viewer = ViewNoisyCirqCirc(cirq_text)
    if args.verb >= 2 :
        print(viewer.gate_aliases()+'\n')
    
    print(f"circuit: {Path(args.name).name}")
    print(viewer.emit_circ(args.lineLen, args.wireLen))


if __name__ == "__main__":
    main()
