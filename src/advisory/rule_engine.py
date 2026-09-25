"""
Agro-Meteorological Advisory Rule Engine
Section 9 of block-to-panchayat-downscaling-spec.md:
Translates downscaled panchayat-level meteorological predictions into actionable farmer advisories.
Phrased matching official IMD Gramin Krishi Mausam Sewa (GKMS) / DAMU bulletin templates.
"""

import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class AdvisorySeverity(str, Enum):
    NORMAL = "Normal"          # Green #6CA02D - Favorable conditions
    ADVISORY = "Advisory"      # Amber #E8720C - Moderate risk, operational caution
    SEVERE = "Severe Alert"    # Red #9B2423 - High risk, immediate protective action


class CropStage(str, Enum):
    SOWING = "Sowing / Germination"
    VEGETATIVE = "Vegetative Growth"
    FLOWERING = "Flowering / Tasseling"
    GRAIN_FILLING = "Grain Filling / Pod Formation"
    MATURITY = "Maturity / Ripening"
    HARVESTING = "Harvesting"


class AgrometAdvisory(BaseModel):
    rule_id: str
    rule_name: str
    severity: AdvisorySeverity
    condition_summary: str
    advisory_text: str
    action_item: str
    pantone_color: str
    hex_color: str


class PanchayatBulletin(BaseModel):
    panchayat_id: str
    panchayat_name: str
    block_name: str
    district_name: str = "Pune"
    forecast_date: str
    rainfall_mm: float
    rainfall_3day_mm: float
    tmax_c: float
    tmin_c: float
    rh_pct: float
    wind_kmh: float
    crop_stage: CropStage
    advisories: List[AgrometAdvisory]
    overall_status: AdvisorySeverity
    gkms_bulletin_text: str


def evaluate_panchayat_advisories(
    panchayat_id: str,
    panchayat_name: str,
    block_name: str,
    forecast_date: str,
    rainfall_mm: float,
    rainfall_3day_mm: float,
    tmax_c: float,
    tmin_c: float,
    rh_pct: float,
    wind_kmh: float,
    crop_stage: CropStage = CropStage.VEGETATIVE,
    crop_name: str = "Soybean / Onion / Sugarcane"
) -> PanchayatBulletin:
    """
    Evaluate downscaled weather metrics against Section 9 agro-meteorological rules.
    Produces a complete GKMS-compliant agromet advisory bulletin.
    """
    advisories: List[AgrometAdvisory] = []
    severities: List[AdvisorySeverity] = []

    # Rule 1: Predicted rainfall < 2.5 mm in next 3 days + crop stage = sowing/vegetative
    # Advisory: Irrigation advisory issued
    if rainfall_3day_mm < 2.5 and crop_stage in [CropStage.SOWING, CropStage.VEGETATIVE]:
        advisories.append(AgrometAdvisory(
            rule_id="RULE_01_IRRIGATION",
            rule_name="Irrigation Advisory",
            severity=AdvisorySeverity.ADVISORY,
            condition_summary=f"3-Day rainfall < 2.5 mm ({rainfall_3day_mm:.1f} mm) during {crop_stage.value}",
            advisory_text=(
                f"Dry spell expected over {panchayat_name} for the next 72 hours. Soil moisture is critical during {crop_stage.value}."
            ),
            action_item=(
                "Apply light to moderate irrigation via drip/sprinkler in morning or evening hours. Mulch with crop residues to conserve root-zone moisture."
            ),
            pantone_color="PMS 1585 C (Advisory Amber)",
            hex_color="#E8720C"
        ))
        severities.append(AdvisorySeverity.ADVISORY)

    # Rule 2: Predicted rainfall > 50 mm in 24 hr
    # Advisory: Waterlogging/drainage advisory
    if rainfall_mm > 50.0:
        advisories.append(AgrometAdvisory(
            rule_id="RULE_02_WATERLOGGING",
            rule_name="Heavy Rain & Waterlogging Advisory",
            severity=AdvisorySeverity.SEVERE,
            condition_summary=f"24-Hour rainfall forecast exceeds 50 mm ({rainfall_mm:.1f} mm)",
            advisory_text=(
                f"Heavy precipitation alert ({rainfall_mm:.1f} mm/24h) for {panchayat_name}. High risk of soil saturation, runoff, and standing water."
            ),
            action_item=(
                "Clear field drainage channels, trenches, and bund outlets immediately to prevent water stagnancy around root zones. Postpone chemical fertilization and intercultural operations."
            ),
            pantone_color="PMS 7621 C (Severe Red)",
            hex_color="#9B2423"
        ))
        severities.append(AdvisorySeverity.SEVERE)

    # Rule 3: Temp > 35°C + humidity > 70%
    # Advisory: Fungal/pest risk flag
    if tmax_c > 35.0 and rh_pct > 70.0:
        advisories.append(AgrometAdvisory(
            rule_id="RULE_03_PEST_FUNGAL",
            rule_name="High Pest & Fungal Disease Risk",
            severity=AdvisorySeverity.ADVISORY,
            condition_summary=f"Max Temp > 35°C ({tmax_c:.1f}°C) with Relative Humidity > 70% ({rh_pct:.1f}%)",
            advisory_text=(
                f"Warm, humid microclimate triggers high vulnerability for fungal blights (Downy Mildew, Anthracnose) and sucking pests in {crop_name}."
            ),
            action_item=(
                "Monitor underside of foliage regularly. Undertake preventive biological or chemical spray (e.g. Copper Oxychloride 2.5 g/L or Trichoderma) once morning dew evaporates."
            ),
            pantone_color="PMS 1585 C (Advisory Amber)",
            hex_color="#E8720C"
        ))
        severities.append(AdvisorySeverity.ADVISORY)

    # Rule 4: Temp < 5°C forecast
    # Advisory: Frost protection advisory
    if tmin_c < 5.0:
        advisories.append(AgrometAdvisory(
            rule_id="RULE_04_FROST",
            rule_name="Frost Protection Advisory",
            severity=AdvisorySeverity.SEVERE,
            condition_summary=f"Minimum Temperature < 5.0°C ({tmin_c:.1f}°C)",
            advisory_text=(
                f"Severe low temperature / ground frost conditions predicted in {panchayat_name}. Risk of leaf cell rupture and flower drop."
            ),
            action_item=(
                "Irrigate fields during late evening to elevate soil temperature. Create protective thatch covers for young seedlings and create smoke screens (smudging) along windward borders."
            ),
            pantone_color="PMS 7621 C (Severe Red)",
            hex_color="#9B2423"
        ))
        severities.append(AdvisorySeverity.SEVERE)

    # Rule 5: Wind speed > 40 km/h forecast
    # Advisory: Crop lodging risk, delay spraying operations
    if wind_kmh > 40.0:
        advisories.append(AgrometAdvisory(
            rule_id="RULE_05_LODGING_WIND",
            rule_name="High Wind & Lodging Risk",
            severity=AdvisorySeverity.SEVERE,
            condition_summary=f"Maximum wind gust > 40 km/h ({wind_kmh:.1f} km/h)",
            advisory_text=(
                f"Strong wind velocity ({wind_kmh:.1f} km/h) forecast over {panchayat_name}. High hazard for crop lodging and heavy spray drift."
            ),
            action_item=(
                "Provide mechanical bamboo staking / earthing-up for sugarcane, banana, and tall vegetables. Strictly suspend all pesticide and foliar spray operations until wind subsides below 15 km/h."
            ),
            pantone_color="PMS 7621 C (Severe Red)",
            hex_color="#9B2423"
        ))
        severities.append(AdvisorySeverity.SEVERE)

    # Determine overall status
    if AdvisorySeverity.SEVERE in severities:
        overall = AdvisorySeverity.SEVERE
    elif AdvisorySeverity.ADVISORY in severities:
        overall = AdvisorySeverity.ADVISORY
    else:
        overall = AdvisorySeverity.NORMAL

    # GKMS Bulletin Phrasing Synthesis
    if not advisories:
        advisory_summary = "Favorable weather conditions prevailing. Normal seasonal intercultural and plant protection activities may proceed."
    else:
        advisory_summary = " ".join([f"[{a.rule_name.upper()}]: {a.action_item}" for a in advisories])

    gkms_text = (
        f"DISTRICT AGROMET FIELD BULLETIN — GRAM PANCHAYAT: {panchayat_name.upper()} (BLOCK: {block_name.upper()})\n"
        f"Issued by GKMS / DAMU Agromet Advisory Cell | Date: {forecast_date}\n"
        f"WEATHER SUMMARY: Rainfall {rainfall_mm:.1f} mm | Tmax {tmax_c:.1f}°C | Tmin {tmin_c:.1f}°C | RH {rh_pct:.1f}% | Wind {wind_kmh:.1f} km/h\n"
        f"STATUS: {overall.value.upper()}\n"
        f"FARMER ADVISORY: {advisory_summary}"
    )

    return PanchayatBulletin(
        panchayat_id=str(panchayat_id),
        panchayat_name=panchayat_name,
        block_name=block_name,
        forecast_date=forecast_date,
        rainfall_mm=round(rainfall_mm, 1),
        rainfall_3day_mm=round(rainfall_3day_mm, 1),
        tmax_c=round(tmax_c, 1),
        tmin_c=round(tmin_c, 1),
        rh_pct=round(rh_pct, 1),
        wind_kmh=round(wind_kmh, 1),
        crop_stage=crop_stage,
        advisories=advisories,
        overall_status=overall,
        gkms_bulletin_text=gkms_text
    )


if __name__ == "__main__":
    # Test sample evaluation
    b = evaluate_panchayat_advisories(
        panchayat_id="185262",
        panchayat_name="AHUPE",
        block_name="AMBEGAON",
        forecast_date="2026-09-25",
        rainfall_mm=62.5,
        rainfall_3day_mm=95.0,
        tmax_c=28.5,
        tmin_c=21.0,
        rh_pct=88.0,
        wind_kmh=42.0,
        crop_stage=CropStage.VEGETATIVE
    )
    print(b.gkms_bulletin_text)
    print("\nAdvisories count:", len(b.advisories))
    for adv in b.advisories:
        print(f" - [{adv.severity.value}] {adv.rule_name}: {adv.condition_summary}")
