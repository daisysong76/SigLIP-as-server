# integrations/google_calendar_integration.py
import asyncio

async def create_event(details: dict) -> str:
    """
    Simulate creating a calendar event.
    In a production system, this function would use the Google Calendar API.
    """
    await asyncio.sleep(0.5)  # Simulate network/API delay
    title = details.get("title", "Unnamed Event")
    time = details.get("time", "Unknown Time")
    return f"Calendar event '{title}' scheduled at {time}."
