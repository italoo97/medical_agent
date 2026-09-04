class GatewayError(Exception):
    """Erro base para qualquer falha originada no gateway."""

    status_code: int = 500

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class UpstreamProviderError(GatewayError):
    """O provedor de LLM (ex.: OpenRouter) respondeu com um erro."""

    status_code = 502
