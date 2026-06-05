from __future__ import annotations

from tools.base import ToolContext, ToolSpec


spec = ToolSpec(
    name="weather",
    description="Get current weather information for a city",
    timeout_seconds=10,
    retryable=True,
    sensitive=False,
)


class WeatherTool:
    spec = spec

    def run(self, input_data, context: ToolContext):
        from pydantic import BaseModel

        class WeatherOutput(BaseModel):
            city: str
            weather: str = "Unknown"
            temperature: str = "N/A"

        return WeatherOutput(city=str(getattr(input_data, "city", "")), weather="N/A", temperature="N/A")
