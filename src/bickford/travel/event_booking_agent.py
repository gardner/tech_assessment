import logging
from smolagents import ToolCallingAgent

import json
import requests
import os
from datetime import datetime
from smolagents import tool
from bickford.travel.config import model
from dotenv import load_dotenv
load_dotenv()

BASE_URL = 'https://api.eventfinda.co.nz'
username = os.getenv("EVENTFINDA_USER")
password = os.getenv("EVENTFINDA_PASS")

logger = logging.getLogger(__name__)

@tool
def find_location_slugs(query: str) -> list[str]:
    """
    Find a location slugs based on a query. Returns a dict of locations slugs with number of events in that location.

    Args:
        query (`str`): query to search for locations.
    Returns:
        `list[str]`: list of location slugs.
    """
    url = BASE_URL + "/v2/locations.json"

    response = requests.get(url, params={"q": query, "venue": "off"}, auth=(username, password))
    response.raise_for_status()
    data = response.json()
    location_slugs = []

    # add slugs in descending order of count_current_events
    for location in sorted(data["locations"], key=lambda x: x["count_current_events"], reverse=True):
        location_slugs.append(location["url_slug"])

    return location_slugs

@tool
def find_event(location_slugs: str, start_date: str, end_date: str) -> str:
    """
    Find an event based on a location slug, start date, and end date (format is ISO 8601, e.g. YYYY-MM-DD)

    Args:
        location_slugs (`str`): list of comma-separated location slugs (e.g. "auckland,auckland-central,auckland-north").
        start_date (`str`): start date of the event (format is ISO 8601, e.g. YYYY-MM-DD).
        end_date (`str`): end date of the event (format is ISO 8601, e.g. YYYY-MM-DD).
    Returns:
        `str`: JSON string of the event.
    """
    url = BASE_URL + "/v2/events.json"
    # try to load ISO 8601 date from start_date
    try:
        # Parse dates and set start to beginning of day, end to end of day
        sdate = datetime.fromisoformat(start_date)
        if sdate.hour == 0 and sdate.minute == 0 and sdate.second == 0:
            # It's just a date, set to start of day
            sdate = sdate.replace(hour=0, minute=0, second=0, microsecond=0)

        edate = datetime.fromisoformat(end_date)
        if edate.hour == 0 and edate.minute == 0 and edate.second == 0:
            # It's just a date, set to end of day (23:59:59.999999)
            edate = edate.replace(hour=23, minute=59, second=59, microsecond=999999)
    except ValueError:
        return "start_date and end_date must be in ISO 8601 format, e.g. YYYY-MM-DD"

    if sdate > edate:
        return "Start date must be before end date"

    params = {
        "location_slug": location_slugs,
        "fields": "event:(url,name,description,is_sold_out,location,sessions),location:(url,name)", #,session:(url,name,start_date,end_date,is_sold_out)
        "start_date": start_date,
        "end_date": end_date,
        "order": "popularity",
        "rows": 5,

    }

    # print(params)

    response = requests.get(
        url,
        params=params,
        auth=(username, password)
    )
    response.raise_for_status()
    data = response.json()
    for event in data["events"]:
        if "sessions" not in event:
            continue

        if  "sessions" not in event["sessions"]:
            continue

        original_sessions = event["sessions"]["sessions"]
        filtered_sessions = []
        for session in original_sessions:
            #Filter sessions by datetime_start and datetime_end
            # Handle case where sessions might be strings or dictionaries
            # Check if session is a dictionary with the expected keys
            if isinstance(session, dict) and "datetime_start" in session and "datetime_end" in session:
                session.pop("session_tickets", None) # remove session_tickets from the session to save space
                try:
                    session_start = datetime.fromisoformat(session["datetime_start"])
                    session_end = datetime.fromisoformat(session["datetime_end"])
                    # Include session if it overlaps with the date range
                    # Session overlaps if: session_start <= edate AND session_end >= sdate
                    if session_start <= edate and session_end >= sdate:
                        filtered_sessions.append(session)
                    else:
                        logger.debug(f"Session {session['datetime_start']} to {session['datetime_end']} does not overlap with range {sdate} to {edate}")
                except (ValueError, KeyError) as e:
                    logger.debug(f"Error parsing session datetime: {e}")
            else:
                logger.debug(f"Session structure issue: {type(session)} - {session if isinstance(session, dict) else 'not a dict'}")

        # Update the event's sessions in the data structure
        event["sessions"]["sessions"] = filtered_sessions

    return json.dumps(data, indent=2)

@tool
def book_event(event_url: str) -> str:
    """
    Book an event based on a URL. Returns a success message.

    Args:
        event_url (`str`): URL of the event to book.
    Returns:
        `str`: JSON string of the success message.
    """
    data = {
        "event_url": event_url,
        "status": "success",
        "ticket_number": "1234567890"
    }
    return json.dumps(data, indent=4)

def create_event_booking_agent():
    return ToolCallingAgent(
        tools=[find_location_slugs, find_event, book_event],
        model=model,
        max_steps=10,
        name="event_booking_agent",
        description="Finds and books events for you.",
    )

if __name__ == "__main__":
    result = find_event("auckland,auckland-central,auckland-north", "2025-12-24", "2025-12-30")
    print(result)
    # agent = create_event_booking_agent()
    # result = agent.run("Find an event in Auckland on 2025-12-30 and book it")
    # print(result)