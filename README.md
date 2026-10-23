# coldplate-loop

Steady-state model of a direct-to-chip (cold-plate) liquid-cooling loop for an AI data-centre hall:
N GPU racks → manifolds → CDU (pump + liquid-to-liquid heat exchanger) → facility water.

**Status:** Step 2 (hydraulics) done. Thermal balances, studies and transients are next.
**All component data are ASSUMED generic values, not vendor data.** Nothing here is a claim about a specific product.

## What it does now

- Pipe losses: Darcy-Weisbach with Colebrook-White friction factor (laminar 64/Re below Re 2300, linear blend to turbulent at 4000 so dp(Q) stays continuous).
- Fittings: dp = K·ρV|V|/2.
- Cold plates, quick disconnects, CDU heat exchanger: power-law dp-flow curves, dp = a·|Q|ⁿ, fitted from published points or built from one rated point plus an exponent.
- Balancing valves from Kv.
- Centrifugal pump: quadratic curve with affinity-law speed scaling.
- Network solver: nodal pressures + edge flows, solved simultaneously (mass balance at nodes, dp-flow relation on edges), with closed-branch support.
- Loop builder: direct-return or reverse-return headers, per-rack branches, blocked-branch case.
- Pump-curve ∩ system-curve operating point and rack flow split.

## Run

```bash
pip install -e ".[dev]"
pytest                          # 17 tests, includes the hand-calc case
python scripts/run_hydraulics.py   # prints operating points, writes results/*.png
```

## Baseline results (10 racks, PG25 at 35 °C, ASSUMED data)

| Case | Total flow [L/min] | Pump rise [kPa] | Rack flow min–max [L/min] |
|---|---|---|---|
| Direct return | 1787 | 197 | 177.3 – 181.9 |
| Reverse return | 1788 | 197 | 178.2 – 179.8 |
| Direct, rack 4 blocked | 1729 | 208 | 190.9 – 195.0 (other racks) |

![pump and system curve](results/pump_system_curve.png)
![rack flow split](results/rack_flow_split.png)

Reading: with the rack branch (~77 kPa at 150 L/min) dominating the pressure budget, the header only adds a few
kPa of imbalance, so direct return spreads flow by ~2.6 % and reverse return by ~0.9 %. Blocking one rack drops the total flow
(pump moves up its curve) while the surviving racks each gain ~7 %.
This ordering is the model's output under these assumptions; the magnitudes depend on the assumed header diameter and rack-branch resistance.

## Validation

**Hand calculation, one rack branch at 150 L/min** (`tests/test_hydraulics.py`, independent arithmetic using Swamee-Jain instead of Colebrook):

| Element | dp [Pa] |
|---|---|
| Hoses, 3 m × 38.1 mm, f = 0.0205, V = 2.19 m/s, Re = 54 900 | 3 930 |
| Hose fittings, K = 4 | 9 737 |
| Rack manifold, K = 2 | 4 868 |
| Quick-disconnect pair (20 kPa at 150 L/min, n = 2) | 20 000 |
| 36 cold plates in parallel, 4.17 L/min each (35 kPa at 4 L/min, n = 1.8) | 37 669 |
| Balancing valve, Kv = 120 | 570 |
| **Total** | **76 773** |

Package result agrees to within 0.2 %. The remaining difference is Swamee-Jain vs Colebrook in the hose friction term (~0.5 % of that term).

Other tests: Hagen-Poiseuille laminar limit, Moody smooth-pipe check, Colebrook vs Haaland, mass conservation, pump curve satisfied at operating point,
affinity laws, reverse-return symmetry, direct-return flow ordering, blocked-branch redistribution, system curve passing through the operating point.

**Not yet done:** comparison of cold-plate/CDU numbers against real vendor or literature data (planned for step 4; replace the assumed values below).

## Assumptions (hydraulic step)

1. Incompressible, steady, single-phase flow. Isothermal at one temperature (35 °C); fluid properties from CoolProp, PG25 = `INCOMP::MPG[0.25]` at 2 bar abs. Real supply/return legs differ by ~10 K, which changes viscosity by ~10 %; the thermal step will evaluate properties per leg.
2. Header ID 156.1 mm (DN150 sch10), 2 m between rack take-offs, one K = 0.5 tee loss per segment (assumed). No elevation changes.
3. Rack branch: 3 m hose ID 38.1 mm, K = 4 for fittings, K = 2 for rack manifold, QD pair 20 kPa at 150 L/min.
4. Cold plate: 35 kPa at 4 L/min per plate, exponent 1.8, 36 plates per rack in parallel with identical flow split. Dp-flow curves are used as given with no viscosity correction.
5. CDU primary-side HX 60 kPa at 1500 L/min (n = 1.85), strainer included.
6. Pump curve through (0, 380), (750, 335), (1500, 250), (2250, 100) L/min–kPa, quadratic fit. Not from a vendor.
7. Pressurisation point (expansion vessel) at pump suction, 2 bar abs.
8. Blocked branch modelled as zero flow; a stopped pump is a quadratic resistance (no check valve).
9. Rack design point is 100 kW at ΔT ≈ 8.4 K (150 L/min); the thermal step will check this.

## Layout

```
src/coldplate_loop/   fluids.py  components.py  network.py  loop.py
tests/                test_hydraulics.py
scripts/              run_hydraulics.py
results/              figures
```

## Roadmap

3. Thermal: rack ΔT, effectiveness-NTU CDU HX, cold-plate thermal resistance vs flow, chip temperature.
4. Validation against vendor/literature data.
5. Studies: IT load sweep, facility water supply temperature sweep, failed pump / blocked branch with thermal consequences.
6. Transients and a reduced-order model.
