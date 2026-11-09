"""Published data used to calibrate and check the model.

Every number here was read from the cited document (23 Oct / 9 Nov 2026) and is
quoted with its stated conditions. Where a document is silent, that is recorded
as None / in the notes rather than filled in.
"""

# --- Vertiv Liebert XDU1350 coolant distribution unit (datasheet SL-70799) -----
XDU1350_SOURCE = ("https://prod.battcon.com/49eb2d/globalassets/shared/"
                  "liebert-xdu1350-coolant-distribution-unitcdu-ds-en-na-sl-70799-web.pdf")
XDU1350 = {
    # Rating basis: facility water 32 C (ASHRAE W3). ATD = approach temperature difference.
    "nominal": {"capacity_kW": 1368.0, "ATD_K": 4.0, "secondary_flow_Lpm": 1200.0,
                "external_dp_kPa": 244.0, "pumps": 2},
    "maximum": {"capacity_kW": 2912.0, "ATD_K": 8.0, "secondary_flow_Lpm": 1800.0,
                "external_dp_kPa": 198.0, "pumps": 3},
    "facility_water_C": 32.0,
    "max_electrical_kW": {2: 13.7, 3: 20.5},   # at maximum flow and external dp
    "primary_dp_kPa": 84.0,                     # at 1200 L/min, 20 % glycol, 27 C (facility side only)
    "primary_flow_Lpm": None,                   # NOT stated in the datasheet
    "notes": ("External dp = pressure available at the CDU secondary ports, net of the CDU's own "
              "internal losses. Datasheet text lists 320 gpm in one place and 317 gpm in the spec table. "
              "One point per pump count: a single curve through both points is an interpolation."),
}

# --- NVIDIA GB200 NVL72 rack via QCT QoolRack datasheet -------------------------
NVL72_SOURCE = ("https://blog.qct.io/wp-content/uploads/2025/04/"
                "QCT-Qoolrack-Stand-Alone_Advanced-Liquid-Cooling-for-NVIDIA-GB200-NVL72-Systems.pdf")
NVL72 = {
    "liquid_load_kW": 115.0,        # plus 17 kW air-cooled, at 130 L/min (conditions not reconciled in source)
    "max_flow_Lpm": 130.0,
    "max_inlet_C": 45.0,
    "max_return_C": 65.0,
    "rack_dp_kPa": None,            # NOT stated
}

# --- Lenovo NVIDIA GB300 NVL72 rack (Lenovo Press LP2357, Table 27 "Water requirements") ---
LENOVO_SOURCE = "https://lenovopress.lenovo.com/lp2357.pdf"
PSI_TO_PA = 6894.757
LENOVO_GB300 = {
    # Required rack flow and rack pressure drop versus fluid supply temperature.
    "supply_C": (25.0, 30.0, 35.0, 40.0, 45.0),
    "flow_Lpm": (59.0, 71.0, 89.0, 119.0, 177.0),
    "rack_dp_psi": (2.3, 3.2, 4.9, 8.5, 18.4),
    "tdp_kW": 135.0,                 # up to 155 kW peak
    "liquid_fraction": 0.90,         # "roughly 90 %" to liquid -> ~121.5 kW; approximate
    "max_inlet_C": 45.0,
    "outlet_C": None,                # NOT stated
    "notes": ("Coolant for the table is not stated (deionized water recommended, PG25 allowed). "
              "Table 18 lists 30 psi static pressure at the QD: a static value, not a dp. "
              "This is the GB300 NVL72; the GB200 NVL72 figures above (QCT) are a different generation."),
}

# --- Cold plate: electronics-cooling.com (Apr 2026), Cu / CuW diamond-pin plates ---
COLD_PLATE_SOURCE = ("https://www.electronics-cooling.com/2026/04/"
                     "high-performance-copper-tungsten-cold-plates-for-direct-to-chip-single-phase-liquid-cooling/")
COLD_PLATE = {
    # 75 x 75 mm calorimeter bar, 1000 W, single-phase water, 1.7-3.8 L/min, design flow 3 L/min.
    # Resistance is bar top surface -> coolant inlet, including a TIM of 1.2 K/kW.
    "R_total_K_per_kW": {"Cu_diamond_pin": 7.6, "CuW_diamond_pin": 10.4},
    "dp_kPa": {"Cu_diamond_pin": 4.15, "CuW_diamond_pin": 4.14},   # flow for these two points NOT stated
    "TIM_K_per_kW": 1.2,
    "design_flow_Lpm": 3.0,
    "chip_power_W": 1000.0,
    "notes": "Laboratory test plates, not a production GB200 plate; one chip class (1 kW) only.",
}

# --- CPC UQD04 quick disconnect --------------------------------------------------
UQD04_SOURCE = ("https://cpcworldwide.com/Portals/0/Downloadable-Content/TRM/Literature/"
                "Spec-Sheets/CPC-Everis-UQD04-Spec-Sheet.pdf")
UQD04 = {"Cv_max": 1.32, "Kv_max": 1.14,
         "notes": "Flow graph not machine-readable; fluid/temperature not stated. UQD04 is a small "
                  "(1/4 in class) coupling, so it does not represent the rack-level disconnects."}
