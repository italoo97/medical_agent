import json

from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from medical_agent.core.config import get_settings

load_dotenv()


def main() -> None:
    settings = get_settings()
    raw = settings.google_service_account_json.get_secret_value()
    info = json.loads(raw)

    print('service account email:', info.get('client_email'))
    print()

    from_info = service_account.Credentials.from_service_account_info
    credentials = from_info(  # type: ignore[no-untyped-call]
        info, scopes=['https://www.googleapis.com/auth/calendar']
    )
    service = build('calendar', 'v3', credentials=credentials)
    response = service.calendarList().list().execute()

    print('--- RAW calendarList response ---')
    print(json.dumps(response, indent=2))


if __name__ == '__main__':
    main()
