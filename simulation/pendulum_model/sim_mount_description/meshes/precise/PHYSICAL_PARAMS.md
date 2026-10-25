# Physical parameters (MuJoCo SI units)

Inertial properties of each rigid body, exported from Fusion 360 and rescaled
to the unit system MuJoCo uses: **metre · kilogram · second · radian**
(length in m, mass in kg, moment of inertia in kg·m²).

Conversion factors applied to the raw Fusion 360 export (g, mm, g·mm²):

| Quantity | Factor |
| --- | --- |
| Mass (g → kg) | ×1e-3 |
| Length / position (mm → m) | ×1e-3 |
| Moment of inertia (g·mm² → kg·m²) | ×1e-9 |

---

## Moror_shaft_precise_assembly v5

- **Mass:** 0.06935 kg
- **Center of mass:** (1e-06, −0.00335, 0.152685) m

**Moment of inertia at center of mass (kg·m²):**

| ixx | iyy | izz | ixy | ixz | iyz |
| --- | --- | --- | --- | --- | --- |
| 5.470821e-05 | 5.540892e-05 | 9.427138e-05 | 1.948e-09 | −2.038e-09 | 2.274404e-06 |

**MuJoCo ready-to-paste:**

```xml
<inertial pos="0.000001 -0.00335 0.152685" mass="0.06935"
           fullinertia="5.470821e-05 5.540892e-05 9.427138e-05 1.948e-09 -2.038e-09 2.274404e-06"/>
```

---

## Pendulum_assembly_precise v3

- **Mass:** 0.019346 kg
- **Center of mass:** (−2.029e-08, 0.095625, 0.14574) m

**Moment of inertia at center of mass (kg·m²):**

| ixx | iyy | izz | ixy | ixz | iyz |
| --- | --- | --- | --- | --- | --- |
| 8.778821e-05 | 4.261985e-05 | 4.550348e-05 | 2.6e-11 | −6.7e-11 | 2.332873e-05 |

**MuJoCo ready-to-paste:**

```xml
<inertial pos="-0.00000002029 0.095625 0.14574" mass="0.019346"
           fullinertia="8.778821e-05 4.261985e-05 4.550348e-05 2.6e-11 -6.7e-11 2.332873e-05"/>
```
