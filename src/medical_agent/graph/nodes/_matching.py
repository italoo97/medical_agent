import difflib
import re

from medical_agent.services.calendar_client import Professional

_FUZZY_MATCH_CUTOFF = 0.75
_TITLE_PATTERN = re.compile(r'\b(dr|dra|doutor|doutora)\b\.?')


def _strip_titles(text: str) -> str:
    return _TITLE_PATTERN.sub('', text).strip()


def _word_matches(word: str, target_word: str) -> bool:
    if word in target_word:
        return True
    return difflib.SequenceMatcher(None, word, target_word).ratio() >= (
        _FUZZY_MATCH_CUTOFF
    )


def _names_match(name: str, professional_name: str) -> bool:
    """Casa nomes com pequenos erros de digitacao (ex.: 'Jhon' e 'John').

    Primeiro tenta substring exata na string inteira -- mais previsivel
    e cobre a maioria dos casos. So recorre a comparacao fuzzy se isso
    falhar, e mesmo assim exige que TODA palavra do nome informado ache
    uma palavra parecida no nome do profissional -- para nao confundir,
    por exemplo, "Jane Roe" com "Jane Doe" so porque o primeiro nome
    bate.
    """
    if name in professional_name:
        return True

    query_words = _strip_titles(name).split()
    target_words = _strip_titles(professional_name).split()
    if not query_words:
        return False

    return all(
        any(_word_matches(word, target_word) for target_word in target_words)
        for word in query_words
    )


def find_professional(
    professionals: list[Professional],
    name: str | None,
    specialty: str | None,
) -> Professional | None:
    for professional in professionals:
        if name and _names_match(name.lower(), professional.name.lower()):
            return professional
        if specialty and specialty.lower() in professional.specialty.lower():
            return professional
    return None
