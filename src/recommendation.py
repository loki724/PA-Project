"""
Evidence-Based Pollution Abatement Recommendation Engine.
Maps detected pollutant concentrations, anomalies, and statistical risk categories
to domain-specific mitigation strategies and environmental control measures.
Supports CO, NOx, NO2, Benzene, PM2.5, PM10, SO2, O3, and multivariate anomalies.
"""

from typing import List, Dict, Any, Optional
import pandas as pd


SCIENTIFIC_DISCLAIMER = (
    "Scientific Disclaimer: Recommendations are derived from observed atmospheric concentration "
    "patterns and established environmental engineering guidelines. Strong statistical correlation "
    "between pollutants and urban activities indicates an environmental pressure, but does not "
    "constitute single-source legal proof without local source-apportionment studies."
)


def get_abatement_recommendations(
    co: Optional[float] = None,
    nox: Optional[float] = None,
    no2: Optional[float] = None,
    benzene: Optional[float] = None,
    pm25: Optional[float] = None,
    pm10: Optional[float] = None,
    so2: Optional[float] = None,
    o3: Optional[float] = None,
    aqi: Optional[float] = None,
    is_anomaly: bool = False,
    risk_category: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generates rule-based, evidence-backed abatement actions for specific observation values.
    Every recommendation is directly tied to a measured pollutant level or detected anomaly.
    """
    recommendations = []

    # 1. Carbon Monoxide (CO) and Nitrogen Oxides (NOx/NO2) - Primary Traffic & Incomplete Combustion
    is_co_elevated = (co is not None and co > 2.2) or (risk_category in ["Elevated", "High"])
    is_co_high = (co is not None and co > 4.2) or (risk_category == "High")
    is_nox_elevated = (nox is not None and nox > 300) or (no2 is not None and no2 > 100)

    if is_co_high or (is_co_elevated and is_nox_elevated):
        recommendations.append({
            "category": "Traffic & Fleet Management",
            "trigger": f"High CO ({co if co is not None else 'N/A'} mg/m³) / Elevated NOx",
            "urgency": "High",
            "actions": [
                "Implement active anti-idling enforcement around primary urban intersections and school zones.",
                "Synchronize traffic light 'green waves' on arterial corridors to reduce stop-and-go driving cycles.",
                "Mandate immediate roadside emissions compliance checks for high-mileage diesel and commercial fleets.",
                "Incentivize off-peak urban logistics and freight deliveries during evening hours."
            ],
            "expected_impact": "Direct reduction in incomplete fuel combustion emissions and peak corridor CO spikes."
        })
    elif is_co_elevated:
        recommendations.append({
            "category": "Traffic Flow Optimization",
            "trigger": f"Elevated CO ({co} mg/m³)",
            "urgency": "Medium",
            "actions": [
                "Encourage public transportation transit usage through temporary park-and-ride incentives.",
                "Promote staggered commercial shift hours to flatten morning and evening peak traffic surges.",
                "Optimize public transit bus lane priority to reduce street congestion."
            ],
            "expected_impact": "Mitigates cumulative vehicle exhaust buildup during peak commute hours."
        })

    # 2. Nitrogen Dioxide (NO2) Specific Actions
    if no2 is not None and no2 > 120:
        recommendations.append({
            "category": "Combustion & Thermal Source Control",
            "trigger": f"Elevated NO2 ({no2} µg/m³)",
            "urgency": "Medium",
            "actions": [
                "Audit local municipal heating systems and commercial boilers for low-NOx burner compliance.",
                "Restrict heavy-duty diesel transport access in sensitive urban residential low-emission zones (LEZs).",
                "Deploy dynamic speed limits on urban highways passing through populated zones."
            ],
            "expected_impact": "Suppresses secondary aerosol formation and precursor oxidants."
        })

    # 3. Particulate Matter (PM2.5 and PM10)
    if (pm25 is not None and pm25 > 35.0) or (pm10 is not None and pm10 > 75.0):
        trigger_str = f"PM2.5: {pm25} µg/m³" if pm25 is not None else f"PM10: {pm10} µg/m³"
        recommendations.append({
            "category": "Particulate Matter Suppression",
            "trigger": f"Elevated Particles ({trigger_str})",
            "urgency": "High",
            "actions": [
                "Mandate misting cannons and dust-suppression curtains at all open construction and demolition sites.",
                "Deploy municipal street sweeping and vacuum trucks with fine particulate filtration.",
                "Prohibit agricultural biomass residue and municipal open waste burning.",
                "Issue advisory notices for sensitive groups to limit prolonged outdoor exertion."
            ],
            "expected_impact": "Rapid mitigation of respirable fine particles and resuspension dust."
        })

    # 4. Sulfur Dioxide (SO2)
    if so2 is not None and so2 > 40.0:
        recommendations.append({
            "category": "Industrial Sulfur Control",
            "trigger": f"Elevated SO2 ({so2} µg/m³)",
            "urgency": "High",
            "actions": [
                "Verify continuous flue-gas desulfurization (FGD) efficiency at nearby power and thermal facilities.",
                "Enforce ultra-low sulfur diesel (ULSD) and marine fuel compliance.",
                "Inspect heavy industrial smelters and petroleum refining units for fugitive acid gas leaks."
            ],
            "expected_impact": "Prevents acid aerosol formation and respiratory irritation."
        })

    # 5. Ground-Level Ozone (O3)
    if o3 is not None and o3 > 100.0:
        recommendations.append({
            "category": "Photochemical Smog Management",
            "trigger": f"Elevated Ozone ({o3} µg/m³)",
            "urgency": "High",
            "actions": [
                "Advise refueling of vehicles after dusk to reduce volatile organic vapor emissions during peak sunlight.",
                "Curtail high-solvent industrial painting and coating operations during midday solar peak hours.",
                "Issue air quality advisories for outdoor athletic events and school sports."
            ],
            "expected_impact": "Suppresses midday photochemical ozone formation cycles."
        })

    # 6. Volatile Organic Compounds: Benzene (C6H6) and Non-Methane Hydrocarbons (NMHC)
    if benzene is not None and benzene > 10.0:
        recommendations.append({
            "category": "Industrial & VOC Containment",
            "trigger": f"Elevated Benzene ({benzene} µg/m³)",
            "urgency": "High",
            "actions": [
                "Inspect Stage II vapor recovery systems at local gasoline service stations for leaks.",
                "Conduct leak detection and repair (LDAR) audits on nearby industrial storage tanks and solvent facilities.",
                "Enforce strict closed-loop handling of organic solvents in dry cleaning and automotive refinishing shops."
            ],
            "expected_impact": "Prevents hazardous aromatic VOC evaporation and suppresses photochemical smog formation."
        })
    elif benzene is not None and benzene > 5.0:
        recommendations.append({
            "category": "VOC Emissions Monitoring",
            "trigger": f"Moderate Benzene ({benzene} µg/m³)",
            "urgency": "Standard",
            "actions": [
                "Increase optical gas imaging audits around chemical transfer points.",
                "Verify compliance with solvent vapor capture thresholds in industrial operations."
            ],
            "expected_impact": "Early mitigation of fugitive aromatic hydrocarbon releases."
        })

    # 7. Multi-Pollutant Isolation Forest Anomalies
    if is_anomaly:
        recommendations.append({
            "category": "Surveillance & Emergency Response",
            "trigger": "Unsupervised Multivariate Isolation Forest Anomaly Flagged",
            "urgency": "High",
            "actions": [
                "Dispatch mobile air monitoring van to triangulate potential anomalous localized emission sources.",
                "Check regional meteorological alerts for atmospheric thermal inversions trapping ground-level pollutants.",
                "Increase sampling frequency of nearby air quality stations from hourly to 15-minute intervals.",
                "Issue precautionary advisories for vulnerable populations (asthma, cardiovascular conditions)."
            ],
            "expected_impact": "Rapid identification of non-standard emission bursts and protective public health measures."
        })

    # 8. Baseline / Normal Condition
    if not recommendations:
        recommendations.append({
            "category": "Routine Maintenance & Preventive Monitoring",
            "trigger": "Pollutant Concentrations Within Normal Baseline",
            "urgency": "Routine",
            "actions": [
                "Maintain scheduled calibration of chemical sensors and reference gas analyzers.",
                "Continue long-term baseline data collection to update seasonal background baselines.",
                "Sustain standard public urban green-space buffers and non-motorized transport lanes."
            ],
            "expected_impact": "Ensures data fidelity and maintains steady ambient air quality."
        })

    return recommendations


def generate_batch_abatement_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes aggregated policy and abatement recommendations across the entire dataset.
    Safely adapts to any available pollutant columns.
    """
    priority_domains = []

    # Check CO
    co_col = "CO(GT)" if "CO(GT)" in df.columns else ("CO" if "CO" in df.columns else None)
    co_high_count = (df[co_col] > 4.2).sum() if co_col is not None else 0

    # Check Benzene
    c6h6_col = "C6H6(GT)" if "C6H6(GT)" in df.columns else ("C6H6" if "C6H6" in df.columns else None)
    c6h6_high_count = (df[c6h6_col] > 10.0).sum() if c6h6_col is not None else 0

    # Check NOx / NO2
    nox_col = "NOx(GT)" if "NOx(GT)" in df.columns else ("NOx" if "NOx" in df.columns else None)
    nox_high_count = (df[nox_col] > 300.0).sum() if nox_col is not None else 0

    # Check PM2.5 / PM10
    pm25_col = "PM2.5" if "PM2.5" in df.columns else ("PM25" if "PM25" in df.columns else None)
    pm25_high_count = (df[pm25_col] > 35.0).sum() if pm25_col is not None else 0

    anom_count = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0

    if co_high_count > 0 or nox_high_count > 0:
        priority_domains.append({
            "domain": "Urban Traffic & Combustion Decoupling",
            "severity": "High Priority",
            "justification": f"Elevated combustion markers detected across multiple observations.",
            "top_action": "Targeted anti-idling zones and traffic signal synchronization on high-congestion corridors."
        })
    if pm25_high_count > 0:
        priority_domains.append({
            "domain": "Respirable Particulate Matter Suppression",
            "severity": "High Priority",
            "justification": f"{pm25_high_count:,} observations exceeded fine particulate (PM2.5) threshold.",
            "top_action": "Construction dust suppression, street washing, and diesel particulate filter mandates."
        })
    if c6h6_high_count > 0:
        priority_domains.append({
            "domain": "VOC & Solvent Vapor Containment",
            "severity": "Medium-to-High Priority",
            "justification": f"{c6h6_high_count:,} observations with Benzene concentrations exceeding 10 µg/m³.",
            "top_action": "Rigorous Stage II vapor recovery inspections at gas stations and solvent-using facilities."
        })
    if anom_count > 0:
        priority_domains.append({
            "domain": "Atmospheric Inversion & Episode Surveillance",
            "severity": "Continuous",
            "justification": f"{anom_count:,} multivariate anomaly states detected by Isolation Forest.",
            "top_action": "Establish dynamic sensor alerts triggered by multi-channel co-elevation."
        })

    if not priority_domains:
        priority_domains.append({
            "domain": "Routine Environmental Baseline Maintenance",
            "severity": "Standard",
            "justification": "Measurements remain largely within typical historical bounds.",
            "top_action": "Continue scheduled sensor maintenance and ambient air surveillance."
        })

    return {
        "priority_domains": priority_domains,
        "co_high_count": int(co_high_count),
        "c6h6_high_count": int(c6h6_high_count),
        "nox_high_count": int(nox_high_count),
        "disclaimer": SCIENTIFIC_DISCLAIMER
    }
