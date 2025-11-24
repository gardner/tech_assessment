# Main orchestration agent
# See https://huggingface.co/docs/smolagents/examples/multiagents

import json
import click
from smolagents import CodeAgent, tool
from bickford.travel.config import model
from bickford.travel.flight_booking_agent import create_flight_booking_agent
from bickford.travel.event_booking_agent import create_event_booking_agent

@tool
def write_itinerary(flight_booking_result: str, event_booking_result: str) -> str:
    """Write an itinerary to a file based on the flight booking and event booking results.
    Args:
        flight_booking_result (`str`): result of the flight booking in JSON format.
        event_booking_result (`str`): result of the event booking in JSON format.
    Returns:
        `str`: itinerary in Markdown format.
    """
    data = {
        "flight_booking_data": flight_booking_result,
        "event_booking_data": event_booking_result
    }
    with open("itinerary.json", "w") as f:
        f.write(json.dumps(data, indent=4))

    return f"SUCCESS: wrote the itinerary to `itinerary.json`"

manager_agent = CodeAgent(
    tools=[write_itinerary],
    model=model,
    managed_agents=[create_flight_booking_agent(), create_event_booking_agent()],
    additional_authorized_imports=["time", "numpy", "pandas", "json"],
)


DEFAULT_PROMPT = "Find a round-trip flight from Sydney to Auckland for 2025-12-30 to 2026-01-05 and book the flight then find an event in Auckland and book a ticket. Be sure to write the itinerary to a file."


@click.command()
@click.argument(
    "prompt",
    required=False,
)
def main(prompt: str | None) -> None:
    """
    Travel booking agent using smolagents CodeAgent.

    Books flights and events based on a natural language prompt.

    PROMPT: Optional natural language prompt for the travel booking task.
    If not provided, uses a default prompt to book a flight and event.
    The prompt should be quoted if it contains spaces.
    """
    prompt_str = prompt if prompt else DEFAULT_PROMPT
    result = manager_agent.run(prompt_str)
    print(result)


if __name__ == "__main__":
    main()
