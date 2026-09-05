import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from medical_agent.services.calendar_client import Professional

SCOPES = ['https://www.googleapis.com/auth/calendar']
_WRITABLE_ROLES = {'writer', 'owner'}


class GoogleCalendarClient:
    def __init__(
        self,
        service_account_json: str,
        timezone: str = 'America/Sao_Paulo',
    ) -> None:
        info = json.loads(service_account_json)
        credentials = service_account.Credentials.from_service_account_info(
            info,
            scopes=SCOPES,
        )
        self._service = build('calendar', 'v3', credentials=credentials)
        self._timezone = ZoneInfo(timezone)

    def _localize(self, moment: datetime) -> datetime:
        """Assume que um datetime sem timezone esta no fuso configurado.

        A API do Google Calendar exige timestamps com timezone explicito;
        a logica de negocio (nodes/appointment_service) trabalha só com
        "horario local", sem se preocupar com fuso -- essa e a unica borda
        onde isso importa.
        """
        if moment.tzinfo is not None:
            return moment
        return moment.replace(tzinfo=self._timezone)

    def list_professionals(self) -> list[Professional]:
        professionals: list[Professional] = []
        page_token: str | None = None

        while True:
            response = (
                self._service.calendarList()
                .list(pageToken=page_token)
                .execute()
            )

            for entry in response.get('items', []):
                if entry.get('accessRole') not in _WRITABLE_ROLES:
                    continue

                professionals.append(
                    Professional(
                        calendar_id=entry['id'],
                        name=entry.get('summary', 'Unknown'),
                        specialty=entry.get('description', 'General'),
                    )
                )

            page_token = response.get('nextPageToken')
            if not page_token:
                break

        return professionals

    def register_calendar(self, calendar_id: str) -> bool:
        try:
            self._service.calendarList().insert(
                body={'id': calendar_id}
            ).execute()
        except HttpError as error:
            if error.resp.status == 409:
                return False
            raise
        return True

    def unregister_calendar(self, calendar_id: str) -> bool:
        try:
            self._service.calendarList().delete(
                calendarId=calendar_id
            ).execute()
        except HttpError as error:
            if error.resp.status == 404:
                return False
            raise
        return True

    def is_available(
        self,
        calendar_id: str,
        start: datetime,
        end: datetime,
    ) -> bool:
        start = self._localize(start)
        end = self._localize(end)

        response = (
            self._service.freebusy()
            .query(
                body={
                    'timeMin': start.isoformat(),
                    'timeMax': end.isoformat(),
                    'items': [{'id': calendar_id}],
                }
            )
            .execute()
        )
        busy_periods = response['calendars'][calendar_id]['busy']
        return len(busy_periods) == 0

    def create_event(
        self,
        calendar_id: str,
        start: datetime,
        end: datetime,
        patient_name: str,
        reason: str,
    ) -> str:
        start = self._localize(start)
        end = self._localize(end)

        event = {
            'summary': f'Consulta — {patient_name}',
            'description': reason,
            'start': {
                'dateTime': start.isoformat(),
                'timeZone': str(self._timezone),
            },
            'end': {
                'dateTime': end.isoformat(),
                'timeZone': str(self._timezone),
            },
            'extendedProperties': {
                'private': {
                    'patientName': patient_name,
                    'reason': reason,
                }
            },
        }
        created = (
            self._service.events()
            .insert(calendarId=calendar_id, body=event)
            .execute()
        )
        return str(created['id'])

    def _find_event_id(
        self,
        calendar_id: str,
        patient_name: str,
        start: datetime,
    ) -> str | None:
        start = self._localize(start)

        response = (
            self._service.events()
            .list(
                calendarId=calendar_id,
                timeMin=(start - timedelta(minutes=1)).isoformat(),
                timeMax=(start + timedelta(minutes=1)).isoformat(),
                privateExtendedProperty=f'patientName={patient_name}',
                singleEvents=True,
            )
            .execute()
        )
        events = response.get('items', [])
        return str(events[0]['id']) if events else None

    def has_appointment(
        self,
        calendar_id: str,
        patient_name: str,
        start: datetime,
    ) -> bool:
        return (
            self._find_event_id(calendar_id, patient_name, start) is not None
        )

    def find_upcoming_appointment(
        self, calendar_id: str, patient_name: str
    ) -> datetime | None:
        now = datetime.now(self._timezone)
        response = (
            self._service.events()
            .list(
                calendarId=calendar_id,
                timeMin=now.isoformat(),
                privateExtendedProperty=f'patientName={patient_name}',
                singleEvents=True,
                orderBy='startTime',
                maxResults=1,
            )
            .execute()
        )
        events = response.get('items', [])
        if not events:
            return None
        return datetime.fromisoformat(events[0]['start']['dateTime'])

    def cancel_event(
        self,
        calendar_id: str,
        patient_name: str,
        start: datetime,
    ) -> None:
        event_id = self._find_event_id(calendar_id, patient_name, start)
        if event_id is None:
            raise LookupError('No matching appointment found to cancel')

        self._service.events().delete(
            calendarId=calendar_id, eventId=event_id
        ).execute()
