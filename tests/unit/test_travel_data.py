"""Unit tests for travel agent data transformation functions."""

import pytest
from bickford.travel.flight_booking_agent import flight_to_dict, result_to_dict
from unittest.mock import Mock
from dataclasses import dataclass, asdict


# Sample dataclass matching fast_flights Flight structure
@dataclass
class MockFlight:
    airline: str
    departure_time: str
    arrival_time: str
    price: float
    is_best: bool
    flight_number: str = "NZ123"


class TestFlightToDict:
    """Test flight_to_dict data transformation."""

    def test_flight_to_dict_basic(self):
        """Test basic flight conversion to dictionary."""
        flight = MockFlight(
            airline="Air New Zealand",
            departure_time="10:00",
            arrival_time="13:00",
            price=299.99,
            is_best=True
        )

        result = flight_to_dict(flight)

        assert result["airline"] == "Air New Zealand"
        assert result["departure_time"] == "10:00"
        assert result["arrival_time"] == "13:00"
        assert result["price"] == 299.99
        assert result["is_best"] is True
        assert result["flight_number"] == "NZ123"

    def test_flight_to_dict_not_best(self):
        """Test flight conversion when is_best is False."""
        flight = MockFlight(
            airline="Qantas",
            departure_time="14:00",
            arrival_time="17:00",
            price=399.99,
            is_best=False,
            flight_number="QF456"
        )

        result = flight_to_dict(flight)

        assert result["is_best"] is False
        assert result["airline"] == "Qantas"


class TestResultToDict:
    """Test result_to_dict data transformation."""

    def test_result_to_dict_basic(self):
        """Test basic result conversion with flights."""
        mock_result = Mock()
        mock_result.current_price = "$500 NZD"
        mock_result.flights = [
            MockFlight("Air NZ", "10:00", "13:00", 299.99, True),
            MockFlight("Qantas", "14:00", "17:00", 399.99, False)
        ]

        result = result_to_dict(mock_result)

        assert result["current_price"] == "$500 NZD"
        assert len(result["flights"]) == 2
        assert result["flights"][0]["airline"] == "Air NZ"
        assert result["flights"][0]["is_best"] is True
        assert result["flights"][1]["airline"] == "Qantas"
        assert result["flights"][1]["is_best"] is False

    def test_result_to_dict_no_flights(self):
        """Test result conversion with empty flights list."""
        mock_result = Mock()
        mock_result.current_price = None
        mock_result.flights = []

        result = result_to_dict(mock_result)

        assert result["current_price"] is None
        assert result["flights"] == []

    def test_result_to_dict_missing_price(self):
        """Test result conversion when price is missing."""
        mock_result = Mock()
        # Simulate missing current_price attribute
        delattr(mock_result, 'current_price') if hasattr(mock_result, 'current_price') else None
        mock_result.flights = [
            MockFlight("Air NZ", "10:00", "13:00", 299.99, True)
        ]

        result = result_to_dict(mock_result)

        # Should handle missing attribute gracefully (returns None via getattr default)
        assert result["current_price"] is None
        assert len(result["flights"]) == 1
