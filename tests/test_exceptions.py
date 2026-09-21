import pytest
from medical_agent.exceptions import GatewayError, UpstreamProviderError


def test_gateway_error_carries_message_and_default_status_code() -> None:
    error = GatewayError('deu ruim')

    assert error.message == 'deu ruim'
    assert error.status_code == 500
    assert str(error) == 'deu ruim'


def test_upstream_provider_error_overrides_status_code() -> None:
    with pytest.raises(UpstreamProviderError) as exc_info:
        raise UpstreamProviderError('openrouter explodiu')

    assert exc_info.value.status_code == 502
    assert exc_info.value.message == 'openrouter explodiu'
