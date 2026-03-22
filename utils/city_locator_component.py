from pathlib import Path

import streamlit.components.v1 as components


_COMPONENT_PATH = Path(__file__).resolve().parent.parent / "components" / "city_locator"
_city_locator = components.declare_component("city_locator", path=str(_COMPONENT_PATH))


def locate_city(key: str = "city_locator"):
    return _city_locator(
        key=key,
        default={
            "status": "idle",
            "latitude": None,
            "longitude": None,
            "error": "",
        },
    )
