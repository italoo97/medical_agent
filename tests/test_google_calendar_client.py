import json
from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
from googleapiclient.errors import HttpError
from medical_agent.services import google_calendar_client as gcc_module
from medical_agent.services.calendar_client import Professional
from medical_agent.services.google_calendar_client import (
    GoogleCalendarClient,
)


class _FakeHttpResponse:
    def __init__(self, status: int) -> None:
        self.status = status
        self.reason = 'error'


def _http_error(status: int) -> HttpError:
    return HttpError(_FakeHttpResponse(status), b'{}')


@pytest.fixture()
def fake_service(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    service = MagicMock()
    monkeypatch.setattr(gcc_module, 'build', lambda *a, **k: service)
    monkeypatch.setattr(
        gcc_module.service_account.Credentials,
        'from_service_account_info',
        staticmethod(lambda info, scopes: 'fake-credentials'),
    )
    return service


@pytest.fixture()
def client(fake_service: MagicMock) -> GoogleCalendarClient:
    return GoogleCalendarClient(
        json.dumps({'type': 'service_account'}),
        timezone='America/Sao_Paulo',
    )


def test_localize_adds_timezone_to_a_naive_datetime(
    client: GoogleCalendarClient,
) -> None:
    naive = datetime(2026, 9, 10, 16, 0)

    localized = client._localize(naive)

    assert localized.tzinfo is not None


def test_localize_keeps_an_already_aware_datetime_unchanged(
    client: GoogleCalendarClient,
) -> None:
    aware = datetime(2026, 9, 10, 16, 0, tzinfo=ZoneInfo('UTC'))

    localized = client._localize(aware)

    assert localized is aware


def test_list_professionals_filters_by_writable_role_and_paginates(
    client: GoogleCalendarClient, fake_service: MagicMock
) -> None:
    page_1 = {
        'items': [
            {
                'id': 'cal-1',
                'summary': 'Dr John',
                'description': 'Cardiology',
                'accessRole': 'owner',
            },
            {
                'id': 'cal-2',
                'summary': 'Reader Only',
                'description': 'x',
                'accessRole': 'reader',
            },
        ],
        'nextPageToken': 'page-2',
    }
    page_2 = {
        'items': [
            {'id': 'cal-3', 'accessRole': 'writer'},
        ],
    }
    execute_mock = (
        fake_service.calendarList.return_value.list.return_value.execute
    )
    execute_mock.side_effect = [page_1, page_2]

    professionals = client.list_professionals()

    assert professionals == [
        Professional('cal-1', 'Dr John', 'Cardiology'),
        Professional('cal-3', 'Unknown', 'General'),
    ]


def test_register_calendar_returns_true_when_newly_registered(
    client: GoogleCalendarClient, fake_service: MagicMock
) -> None:
    insert_mock = fake_service.calendarList.return_value.insert
    insert_mock.return_value.execute.return_value = {}

    assert client.register_calendar('cal-1') is True


def test_register_calendar_returns_false_when_already_registered(
    client: GoogleCalendarClient, fake_service: MagicMock
) -> None:
    insert_mock = fake_service.calendarList.return_value.insert
    insert_mock.return_value.execute.side_effect = _http_error(409)

    assert client.register_calendar('cal-1') is False


def test_register_calendar_reraises_other_http_errors(
    client: GoogleCalendarClient, fake_service: MagicMock
) -> None:
    insert_mock = fake_service.calendarList.return_value.insert
    insert_mock.return_value.execute.side_effect = _http_error(500)

    with pytest.raises(HttpError):
        client.register_calendar('cal-1')


def test_unregister_calendar_returns_true_when_removed(
    client: GoogleCalendarClient, fake_service: MagicMock
) -> None:
    delete_mock = fake_service.calendarList.return_value.delete
    delete_mock.return_value.execute.return_value = {}

    assert client.unregister_calendar('cal-1') is True


def test_unregister_calendar_returns_false_when_not_registered(
    client: GoogleCalendarClient, fake_service: MagicMock
) -> None:
    delete_mock = fake_service.calendarList.return_value.delete
    delete_mock.return_value.execute.side_effect = _http_error(404)

    assert client.unregister_calendar('cal-1') is False


def test_unregister_calendar_reraises_other_http_errors(
    client: GoogleCalendarClient, fake_service: MagicMock
) -> None:
    delete_mock = fake_service.calendarList.return_value.delete
    delete_mock.return_value.execute.side_effect = _http_error(500)

    with pytest.raises(HttpError):
        client.unregister_calendar('cal-1')


def test_is_available_true_when_no_busy_periods(
    client: GoogleCalendarClient, fake_service: MagicMock
) -> None:
    query_mock = fake_service.freebusy.return_value.query
    query_mock.return_value.execute.return_value = {
        'calendars': {'cal-1': {'busy': []}}
    }

    available = client.is_available(
        'cal-1', datetime(2026, 9, 10, 16, 0), datetime(2026, 9, 10, 16, 30)
    )

    assert available is True


def test_is_available_false_when_there_is_a_busy_period(
    client: GoogleCalendarClient, fake_service: MagicMock
) -> None:
    query_mock = fake_service.freebusy.return_value.query
    query_mock.return_value.execute.return_value = {
        'calendars': {
            'cal-1': {
                'busy': [
                    {
                        'start': '2026-09-10T16:00:00',
                        'end': '2026-09-10T16:30:00',
                    }
                ]
            }
        }
    }

    available = client.is_available(
        'cal-1', datetime(2026, 9, 10, 16, 0), datetime(2026, 9, 10, 16, 30)
    )

    assert available is False


def test_create_event_returns_the_new_event_id(
    client: GoogleCalendarClient, fake_service: MagicMock
) -> None:
    insert_mock = fake_service.events.return_value.insert
    insert_mock.return_value.execute.return_value = {'id': 'event-123'}

    event_id = client.create_event(
        'cal-1',
        datetime(2026, 9, 10, 16, 0),
        datetime(2026, 9, 10, 16, 30),
        'Maria Santos',
        'check-up',
    )

    assert event_id == 'event-123'


def test_has_appointment_true_when_a_matching_event_exists(
    client: GoogleCalendarClient, fake_service: MagicMock
) -> None:
    fake_service.events.return_value.list.return_value.execute.return_value = {
        'items': [{'id': 'event-123'}]
    }

    assert (
        client.has_appointment(
            'cal-1', 'Maria Santos', datetime(2026, 9, 10, 16, 0)
        )
        is True
    )


def test_has_appointment_false_when_nothing_matches(
    client: GoogleCalendarClient, fake_service: MagicMock
) -> None:
    fake_service.events.return_value.list.return_value.execute.return_value = {
        'items': []
    }

    assert (
        client.has_appointment(
            'cal-1', 'Maria Santos', datetime(2026, 9, 10, 16, 0)
        )
        is False
    )


def test_find_upcoming_appointment_returns_the_start_datetime(
    client: GoogleCalendarClient, fake_service: MagicMock
) -> None:
    fake_service.events.return_value.list.return_value.execute.return_value = {
        'items': [{'start': {'dateTime': '2026-09-10T16:00:00-03:00'}}]
    }

    start = client.find_upcoming_appointment('cal-1', 'Maria Santos')

    assert start == datetime.fromisoformat('2026-09-10T16:00:00-03:00')


def test_find_upcoming_appointment_returns_none_when_nothing_found(
    client: GoogleCalendarClient, fake_service: MagicMock
) -> None:
    fake_service.events.return_value.list.return_value.execute.return_value = {
        'items': []
    }

    assert client.find_upcoming_appointment('cal-1', 'Maria Santos') is None


def test_cancel_event_deletes_the_matching_event(
    client: GoogleCalendarClient, fake_service: MagicMock
) -> None:
    fake_service.events.return_value.list.return_value.execute.return_value = {
        'items': [{'id': 'event-123'}]
    }
    delete_execute = (
        fake_service.events.return_value.delete.return_value.execute
    )

    client.cancel_event('cal-1', 'Maria Santos', datetime(2026, 9, 10, 16, 0))

    fake_service.events.return_value.delete.assert_called_with(
        calendarId='cal-1', eventId='event-123'
    )
    delete_execute.assert_called_once()


def test_cancel_event_raises_lookup_error_when_nothing_matches(
    client: GoogleCalendarClient, fake_service: MagicMock
) -> None:
    fake_service.events.return_value.list.return_value.execute.return_value = {
        'items': []
    }

    with pytest.raises(LookupError):
        client.cancel_event(
            'cal-1', 'Maria Santos', datetime(2026, 9, 10, 16, 0)
        )
