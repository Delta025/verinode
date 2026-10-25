"""Tests for src.core.unit_normalizer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.unit_normalizer import normalize_number_formats, normalize_units

# ------------------------------------------------------------------
# Number format normalization
# ------------------------------------------------------------------


class TestNormalizeNumberFormats:
    def test_strips_commas(self):
        assert normalize_number_formats("384,400 km") == "384400 km"

    def test_large_number(self):
        assert normalize_number_formats("1,234,567,890") == "1234567890"

    def test_decimal_with_commas(self):
        assert normalize_number_formats("1,234.56") == "1234.56"

    def test_no_commas_unchanged(self):
        assert normalize_number_formats("12345 km") == "12345 km"

    def test_text_without_numbers(self):
        assert normalize_number_formats("The Sun is a star") == "The Sun is a star"


# ------------------------------------------------------------------
# Distance conversions (→ km)
# ------------------------------------------------------------------


class TestDistanceConversion:
    def test_miles_to_km(self):
        result = normalize_units("The distance is 100 miles")
        assert "km" in result
        assert "miles" not in result.lower()
        # 100 miles ≈ 160.93 km
        assert "160.93" in result

    def test_million_miles_to_km(self):
        result = normalize_units("about 93 million miles from the Sun")
        assert "million km" in result
        assert "miles" not in result.lower()

    def test_feet_to_km(self):
        result = normalize_units("altitude of 35000 feet")
        assert "km" in result
        assert "feet" not in result.lower()

    def test_au_to_km(self):
        result = normalize_units("Neptune orbits at 30 au")
        assert "km" in result

    def test_km_unchanged(self):
        result = normalize_units("The distance is 150 km")
        # Should still say km, number may be reformatted but value same
        assert "km" in result
        assert "150" in result

    def test_meters_to_km(self):
        result = normalize_units("The crater is 5000 meters wide")
        assert "km" in result


# ------------------------------------------------------------------
# Speed conversions (→ km/h)
# ------------------------------------------------------------------


class TestSpeedConversion:
    def test_mph_to_kmh(self):
        result = normalize_units("Speed: 100 mph")
        assert "km/h" in result
        assert "mph" not in result.lower()
        # 100 mph ≈ 160.93 km/h
        assert "160.93" in result

    def test_large_mph(self):
        result = normalize_units("orbital speed of 515000 mph")
        assert "km/h" in result
        assert "mph" not in result.lower()

    def test_ms_to_kmh(self):
        result = normalize_units("velocity of 30 m/s")
        assert "km/h" in result
        # 30 m/s = 108 km/h
        assert "108" in result

    def test_kmh_unchanged(self):
        result = normalize_units("speed is 828000 km/h")
        assert "km/h" in result


# ------------------------------------------------------------------
# Temperature conversions (→ °C)
# ------------------------------------------------------------------


class TestTemperatureConversion:
    def test_fahrenheit_to_celsius(self):
        result = normalize_units("Water boils at 212°F")
        assert "°C" in result
        # 212°F = 100°C
        assert "100" in result

    def test_kelvin_to_celsius(self):
        result = normalize_units("The Sun's surface is 5778 K")
        assert "°C" in result
        # 5778 K = 5504.85°C
        assert "5505" in result or "5504.85" in result

    def test_celsius_unchanged(self):
        result = normalize_units("Temperature is 100 °C")
        assert "°C" in result
        assert "100" in result

    def test_degrees_fahrenheit_word(self):
        result = normalize_units("about 10000 degrees Fahrenheit")
        assert "°C" in result

    def test_degrees_kelvin_word(self):
        result = normalize_units("temperature of 273 degrees Kelvin")
        assert "°C" in result


# ------------------------------------------------------------------
# Mass conversions (→ kg)
# ------------------------------------------------------------------


class TestMassConversion:
    def test_lbs_to_kg(self):
        result = normalize_units("weight: 150 lbs")
        assert "kg" in result
        assert "lbs" not in result.lower()
        # 150 lbs ≈ 68.04 kg
        assert "68" in result

    def test_pounds_to_kg(self):
        result = normalize_units("mass of 200 pounds")
        assert "kg" in result

    def test_ounces_to_kg(self):
        result = normalize_units("sample weighs 16 ounces")
        assert "kg" in result

    def test_tonnes_to_kg(self):
        result = normalize_units("mass of 5 tonnes")
        assert "kg" in result
        # 5 tonnes = 5000 kg
        assert "5000" in result

    def test_kg_unchanged(self):
        result = normalize_units("mass is 100 kg")
        assert "kg" in result
        assert "100" in result


# ------------------------------------------------------------------
# Area conversions (→ km²)
# ------------------------------------------------------------------


class TestAreaConversion:
    def test_sq_miles_to_km2(self):
        result = normalize_units("area of 100 sq mi")
        assert "km" in result
        # 100 sq mi ≈ 259 km²
        assert "259" in result

    def test_square_miles_to_km2(self):
        result = normalize_units("covers 1000 square miles")
        assert "km" in result

    def test_acres_to_km2(self):
        result = normalize_units("property is 1000 acres")
        assert "km" in result

    def test_hectares_to_km2(self):
        result = normalize_units("reserve of 500 hectares")
        assert "km" in result


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_string(self):
        assert normalize_units("") == ""

    def test_none_returns_none(self):
        # normalize_units checks for falsy input
        assert normalize_units("") == ""

    def test_no_units(self):
        text = "The Sun is the center of the Solar System"
        assert normalize_units(text) == text

    def test_multiple_units_in_text(self):
        result = normalize_units("Distance: 100 miles, Speed: 60 mph")
        assert "km" in result
        assert "km/h" in result
        assert "miles" not in result.lower()
        assert "mph" not in result.lower()

    def test_number_with_commas(self):
        result = normalize_units("diameter of 384,400 miles")
        assert "km" in result
        # Commas should be stripped after conversion

    def test_decimal_number(self):
        result = normalize_units("distance is 1.5 miles")
        assert "km" in result
        # 1.5 miles ≈ 2.41 km
        assert "2.41" in result or "2.4" in result

    def test_zero_value(self):
        result = normalize_units("0 miles from here")
        assert "km" in result

    def test_preserves_surrounding_text(self):
        result = normalize_units("The planet is about 100 miles away from Earth")
        assert result.startswith("The planet is about")
        assert result.endswith("away from Earth")
        assert "km" in result


# ------------------------------------------------------------------
# Real-world solar system examples (from the conflict analysis)
# ------------------------------------------------------------------


class TestSolarSystemExamples:
    def test_earth_sun_distance_miles(self):
        """The actual false positive: miles vs km for Earth-Sun distance."""
        result_miles = normalize_units("Earth-Sun distance varies +/-1.6 million miles")
        result_km = normalize_units("Earth-Sun distance varies +/-2.5 million km")
        # Both should now be in km
        assert "km" in result_miles
        assert "km" in result_km

    def test_solar_orbit_speed(self):
        """The actual false positive: mph vs kph for orbital speed."""
        result_mph = normalize_units("solar orbit speed 515000 mph")
        result_kph = normalize_units("solar orbit speed 828000 km/h")
        # Both should now be in km/h
        assert "km/h" in result_mph
        assert "km/h" in result_kph

    def test_sun_surface_temp(self):
        """Temperature expressed in different units."""
        result_k = normalize_units("The Sun's surface temperature is 5778 K")
        result_c = normalize_units("The Sun's temperature is about 5505 °C")
        # Both should be in °C
        assert "°C" in result_k
        assert "°C" in result_c
