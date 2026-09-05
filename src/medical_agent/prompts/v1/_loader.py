from pathlib import Path

_TEMPLATES_DIR = Path(__file__).parent / 'templates'


def load_sections(filename: str) -> dict[str, str]:
    """Lê um .md em templates/ e separa suas seções `### nome`."""
    content = (_TEMPLATES_DIR / filename).read_text(encoding='utf-8')

    sections: dict[str, str] = {}
    current_name: str | None = None
    current_lines: list[str] = []

    for line in content.splitlines():
        if line.startswith('### '):
            if current_name is not None:
                sections[current_name] = '\n'.join(current_lines).strip()
            current_name = line.removeprefix('### ').strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_name is not None:
        sections[current_name] = '\n'.join(current_lines).strip()

    return sections
