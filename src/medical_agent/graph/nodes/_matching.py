from medical_agent.services.calendar_client import Professional


def find_professional(
    professionals: list[Professional],
    name: str | None,
    specialty: str | None,
) -> Professional | None:
    for professional in professionals:
        if name and name.lower() in professional.name.lower():
            return professional
        if specialty and specialty.lower() in professional.specialty.lower():
            return professional
    return None
