import json
from datetime import datetime, timedelta

from google.oauth2 import service_account
from googleapiclient.discovery import build

from medical_agent.services.calendar_client import Professional

SCOPES = ['https://www.googleapis.com/auth/calendar']
_WRITABLE_ROLES = {'writer', 'owner'}


class GoogleCalendarClient:
    def __init__(self, service_account_json: str) -> None:
        info = json.loads(service_account_json)
        credentials = service_account.Credentials.from_service_account_info(
            info,
            scopes=SCOPES,
        )
        self._service = build('calendar', 'v3', credentials=credentials)

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

    def is_available(
        self,
        calendar_id: str,
        start: datetime,
        end: datetime,
    ) -> bool:
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
        event = {
            'summary': f'Consulta — {patient_name}',
            'description': reason,
            'start': {'dateTime': start.isoformat()},
            'end': {'dateTime': end.isoformat()},
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

    def cancel_event(
        self,
        calendar_id: str,
        patient_name: str,
        start: datetime,
    ) -> None:
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
        if not events:
            raise LookupError('No matching appointment found to cancel')

        self._service.events().delete(
            calendarId=calendar_id, eventId=events[0]['id']
        ).execute()
