
import requests


class WeatherService:

    def get_coordinates(self, city: str) -> tuple[float, float]:
        """
        Obtiene las coordenadas de una ciudad

        Args:
            city (str): Nombre de la ciudad
        Returns:
            tuple: (latitud, longitud)
        """
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {
            "name": city,
            "count": 1,
            "language": "es"
        }

        response = requests.get(url, params=params)
        data = response.json()

        if not data.get("results"):
            return None, None

        result = data["results"][0]
        return result["latitude"], result["longitude"]

    def get_current_weather_by_city(self, city: str) -> dict[str, any]:
        """
        Obtiene el clima actual de una ciudad

        Args:
            city (str): Nombre de la ciudad

        Returns:
            dict: Datos del clima
        """
        lat, lon = self.get_coordinates(city)

        if lat is None:
            return {"error": "Ciudad no encontrada"}

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
            "timezone": "America/Sao_Paulo"
        }

        response = requests.get(url, params=params)

        data = response.json()

        return data.get("current_weather", {})
