from contextvars import ContextVar


# 用 ContextVar 保存“当前请求”的用户上下文，避免工具函数层层传参。
_user_city_var: ContextVar[str] = ContextVar("user_city", default="")
_user_weather_var: ContextVar[str] = ContextVar("user_weather", default="")
_user_temperature_var: ContextVar[str] = ContextVar("user_temperature", default="")
_user_id_var: ContextVar[str] = ContextVar("user_id", default="")


def set_user_city(city: str) -> None:
    _user_city_var.set((city or "").strip())


def get_user_city() -> str:
    return _user_city_var.get().strip()


def set_user_weather(weather: str) -> None:
    _user_weather_var.set((weather or "").strip())


def get_user_weather() -> str:
    return _user_weather_var.get().strip()


def set_user_temperature(temperature: str) -> None:
    _user_temperature_var.set((temperature or "").strip())


def get_user_temperature() -> str:
    return _user_temperature_var.get().strip()


def set_user_id(user_id: str) -> None:
    _user_id_var.set((user_id or "").strip())


def get_user_id() -> str:
    return _user_id_var.get().strip()
