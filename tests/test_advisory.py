"""
Test GKMS Agro-Meteorological Advisory Rule Engine
Matches Section 9 of block-to-panchayat-downscaling-spec.md.
"""

import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pytest
from src.advisory.rule_engine import (
    evaluate_panchayat_advisories,
    CropStage,
    AdvisorySeverity
)


def test_rule_1_irrigation():
    # Condition: 3-day rainfall < 2.5 mm + vegetative stage -> Irrigation advisory
    b = evaluate_panchayat_advisories(
        panchayat_id="101",
        panchayat_name="BARAMATI RURAL",
        block_name="BARAMATI",
        forecast_date="2026-09-25",
        rainfall_mm=0.0,
        rainfall_3day_mm=1.2,
        tmax_c=31.0,
        tmin_c=22.0,
        rh_pct=55.0,
        wind_kmh=12.0,
        crop_stage=CropStage.VEGETATIVE
    )
    rule_ids = [a.rule_id for a in b.advisories]
    assert "RULE_01_IRRIGATION" in rule_ids
    assert b.overall_status in [AdvisorySeverity.ADVISORY, AdvisorySeverity.SEVERE]


def test_rule_2_waterlogging():
    # Condition: 24h rainfall > 50 mm -> Waterlogging/drainage advisory
    b = evaluate_panchayat_advisories(
        panchayat_id="102",
        panchayat_name="LONAVALA WEST",
        block_name="MAVAL",
        forecast_date="2026-09-25",
        rainfall_mm=68.0,
        rainfall_3day_mm=120.0,
        tmax_c=24.0,
        tmin_c=19.0,
        rh_pct=95.0,
        wind_kmh=18.0
    )
    rule_ids = [a.rule_id for a in b.advisories]
    assert "RULE_02_WATERLOGGING" in rule_ids
    assert b.overall_status == AdvisorySeverity.SEVERE


def test_rule_3_pest_fungal():
    # Condition: Temp > 35°C + RH > 70% -> Fungal/pest risk flag
    b = evaluate_panchayat_advisories(
        panchayat_id="103",
        panchayat_name="INDAPUR CENTRAL",
        block_name="INDAPUR",
        forecast_date="2026-09-25",
        rainfall_mm=5.0,
        rainfall_3day_mm=15.0,
        tmax_c=36.5,
        tmin_c=25.0,
        rh_pct=76.0,
        wind_kmh=14.0
    )
    rule_ids = [a.rule_id for a in b.advisories]
    assert "RULE_03_PEST_FUNGAL" in rule_ids


def test_rule_4_frost():
    # Condition: Temp < 5°C -> Frost protection advisory
    b = evaluate_panchayat_advisories(
        panchayat_id="104",
        panchayat_name="SHIVNERI HIGH",
        block_name="JUNNAR",
        forecast_date="2026-12-25",
        rainfall_mm=0.0,
        rainfall_3day_mm=0.0,
        tmax_c=22.0,
        tmin_c=4.2,
        rh_pct=50.0,
        wind_kmh=8.0
    )
    rule_ids = [a.rule_id for a in b.advisories]
    assert "RULE_04_FROST" in rule_ids
    assert b.overall_status == AdvisorySeverity.SEVERE


def test_rule_5_lodging_wind():
    # Condition: Wind > 40 km/h -> Lodging risk, delay spraying
    b = evaluate_panchayat_advisories(
        panchayat_id="105",
        panchayat_name="TORNA HILL",
        block_name="VELHE",
        forecast_date="2026-07-15",
        rainfall_mm=25.0,
        rainfall_3day_mm=60.0,
        tmax_c=26.0,
        tmin_c=20.0,
        rh_pct=85.0,
        wind_kmh=48.5
    )
    rule_ids = [a.rule_id for a in b.advisories]
    assert "RULE_05_LODGING_WIND" in rule_ids
    assert b.overall_status == AdvisorySeverity.SEVERE
