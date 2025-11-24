from pprint import pprint
from smolagents import ToolCallingAgent

import json
from smolagents import tool
from dataclasses import asdict
from fast_flights import FlightData, Passengers, Result, get_flights, search_airport, FlightData, Passengers, create_filter, get_flights_from_filter
from bickford.travel.config import model, logger

from dotenv import load_dotenv
load_dotenv()

# Helper functions to convert Result and Flight objects to dictionaries

def flight_to_dict(flight):
    return asdict(flight)

def result_to_dict(result):
    return {
        "current_price": getattr(result, 'current_price', None),
        "flights": [flight_to_dict(flight) for flight in getattr(result, 'flights', [])]
    }

# Agent tools

@tool
def find_airport_codes(city_name: str) -> str:
    """Find 3 letter airport code for a city name.

    Args:
        city_name (`str`): city name (e.g. Auckland).
    Returns:
        `str`: 3 letter airport code (e.g. AKL)."""

    airports = search_airport(city_name)
    return airports[0].value

@tool
def find_one_way_flight(from_airport: str, to_airport: str, date: str) -> str:
    """Find one-way flights between two airports on a given date

    Args:
        from_airport (`str`): 3 letter airport code of the departure airport (e.g. AKL).
        to_airport (`str`): 3 letter airport code of the arrival airport (e.g. SYD).
        date (`str`): date of the flight (formatted like 2025-01-01).
    Returns:
        `str`: result of the flight search in JSON format."""

    result: Result = get_flights(
        flight_data=[
            FlightData(date=date, from_airport=from_airport, to_airport=to_airport)
        ],
        trip="one-way",
        seat="economy",
        passengers=Passengers(adults=2, children=1, infants_in_seat=0, infants_on_lap=0),
        fetch_mode="fallback",
    )

    result = result_to_dict(result)

    data = {
        "current_price": result.get("current_price", None),
        "flights": [flight for flight in result.get("flights", []) if flight.get("is_best", False) is True],
    }

    return json.dumps(data, indent=4)

@tool
def find_round_trip_flight(from_airport: str, to_airport: str, departure_date: str, return_date: str) -> str:
    """Find round-trip flights between two airports on a given dates

    Args:
        from_airport (`str`): 3 letter airport code of the departure airport (e.g. AKL).
        to_airport (`str`): 3 letter airport code of the arrival airport (e.g. SYD).
        departure_date (`str`): date of the departure flight (e.g. 2025-12-30).
        return_date (`str`): date of the return flight (e.g. 2026-01-29).
    Returns:
        `str`: result of the flight search in JSON format."""

    filter = create_filter(
        flight_data=[
            FlightData(date=departure_date, from_airport=from_airport, to_airport=to_airport),
            FlightData(date=return_date, from_airport=to_airport, to_airport=from_airport),
        ],
        trip="round-trip",
        seat="economy",
        passengers=Passengers(adults=2, children=0, infants_in_seat=0, infants_on_lap=0),
    )

    b64 = filter.as_b64().decode('utf-8')
    booking_url = f"https://www.google.com/travel/flights?tfs={b64}"
    result = get_flights_from_filter(
        filter,
        mode="fallback",
    )

    result = result_to_dict(result)

    data = {
        "current_price": result.get("current_price", None),
        "flights": [flight for flight in result.get("flights", []) if flight.get("is_best", False) is True],
        "booking_url": booking_url,
    }
    return json.dumps(data, indent=4)

@tool
def book_flight(booking_url: str) -> str:
    """Book a flight based on a URL. Returns a success message.

    Args:
        booking_url (`str`): URL of the flight to book (e.g. https://www.flighthub.com/flight/AKL-SYD/2025-12-30).
    Returns:
        `str`: success message."""
    data = {
        "booking_url": booking_url,
        "status": "success",
        "ticket_number": "1234567890"
    }
    return json.dumps(data, indent=4)


def create_flight_booking_agent():
    return ToolCallingAgent(
        tools=[find_airport_codes, find_one_way_flight, find_round_trip_flight, book_flight],
        model=model,
        max_steps=20,
        name="flight_booking_agent",
        description="Finds and books flights for you.",
    )
if __name__ == "__main__":
    agent = create_flight_booking_agent()

    result = agent.run("Find a round-trip flight from Auckland to Sydney for 2025-12-30 to 2026-01-29 and book the flight")

    logger.info(result)