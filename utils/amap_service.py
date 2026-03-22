import os
from typing import Any

import requests


AMAP_BASE_URL = "https://restapi.amap.com"
REQUEST_TIMEOUT = 10


class AMapServiceError(RuntimeError):
    pass


def get_amap_api_key() -> str:
    api_key = (os.getenv("AMAP_API_KEY") or "").strip()
    if not api_key:
        raise AMapServiceError("未读取到 AMAP_API_KEY，请重启 Streamlit 或终端后重试。")
    return api_key


def _request_json(path: str, params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(
        f"{AMAP_BASE_URL}{path}",
        params={**params, "key": get_amap_api_key(), "output": "JSON"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("status") != "1":
        raise AMapServiceError(data.get("info") or "高德接口调用失败。")
    return data


def _normalize_city(city_value: Any, district: str, province: str) -> str:
    if isinstance(city_value, list):
        city = "".join(item for item in city_value if isinstance(item, str)).strip()
    else:
        city = str(city_value or "").strip()

    if city:
        return city
    if province.endswith("市"):
        return province
    if district:
        return district
    return province


def reverse_geocode(latitude: float, longitude: float) -> dict[str, str]:
    data = _request_json(
        "/v3/geocode/regeo",
        {
            "location": f"{longitude},{latitude}",
            "extensions": "base",
            "radius": 1000,
            "roadlevel": 0,
        },
    )

    address_component = data.get("regeocode", {}).get("addressComponent", {})
    province = str(address_component.get("province") or "").strip()
    district = str(address_component.get("district") or "").strip()
    adcode = str(address_component.get("adcode") or "").strip()
    city = _normalize_city(address_component.get("city"), district, province)

    if not city:
        raise AMapServiceError("高德逆地理编码未返回城市信息。")

    return {
        "city": city,
        "province": province,
        "district": district,
        "adcode": adcode,
    }


def get_live_weather(city_code: str) -> dict[str, str]:
    if not city_code:
        raise AMapServiceError("缺少城市编码，无法获取天气。")

    data = _request_json(
        "/v3/weather/weatherInfo",
        {
            "city": city_code,
            "extensions": "base",
        },
    )

    lives = data.get("lives") or []
    if not lives:
        raise AMapServiceError("高德天气接口未返回实时天气信息。")

    live = lives[0]
    weather = str(live.get("weather") or "").strip()
    temperature = str(live.get("temperature") or "").strip()
    if not weather:
        raise AMapServiceError("高德天气接口返回的天气字段为空。")
    if not temperature:
        raise AMapServiceError("高德天气接口返回的温度字段为空。")
    return {
        "weather": weather,
        "temperature": temperature,
    }


def get_location_snapshot(latitude: float, longitude: float) -> dict[str, str]:
    location = reverse_geocode(latitude, longitude)
    weather_info = get_live_weather(location["adcode"])
    return {
        "city": location["city"],
        "province": location["province"],
        "district": location["district"],
        "adcode": location["adcode"],
        "weather": weather_info["weather"],
        "temperature": weather_info["temperature"],
    }
