"""
API RP 985 Refractory Thermal Calculator (mobile-friendly visual edition)
Run locally with: streamlit run app.py
"""
from math import exp, log, pi, sqrt
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# 1. CORE API RP 985 CALCULATION ENGINE (unchanged)
# ==========================================
SIGMA = 0.1714e-8
G = 32.174 * 3600**2

Z = [(500, 23.60), (1000, 14.85), (1500, 14.13), (2000, 15.61)]
IC = [(500, 3.32), (1000, 3.02), (1500, 3.24), (2000, 3.60)]
IB = [(500, 0.57), (1000, 0.80), (1500, 1.13), (2000, 1.60)]

LININGS = {
    1: ("Lining 1 · 1 layer (Castable X)", [("Castable X", 6.0, [(0, 4.0)])]),
    2: ("Lining 2 · 1 layer (Castable Y)", [("Castable Y", 6.0, [(500, 0.25), (1000, 0.50), (1500, 1.0), (2000, 2.0)])]),
    3: ("Lining 3 · 2 layers (Castable Z + Insulating Cast)", [("Castable Z", 6.0, Z), ("Insulating Cast", 4.0, IC)]),
    4: ("Lining 4 · 3 layers (Castable Z + Insul. Cast + Board)", [("Castable Z", 6.0, Z), ("Insulating Cast", 4.0, IC), ("Insulation board", 1.5, IB)]),
    5: ("Lining 5 · 3 layers (extended to 2500 °F)", [
        ("Castable Z ext", 6.0, [(500, 23.60), (1000, 14.85), (2000, 15.61), (2500, 19.0)]),
        ("Insulating Cast ext", 4.0, [(0, 4.05), (1000, 3.02), (2000, 3.60), (2500, 4.0)]),
        ("Insulation board ext", 1.5, [(0, 0.36), (500, 0.57), (1000, 0.80), (2000, 1.60)]),
    ]),
}

# (geom, d_dict, lining_id, thick_list, eps, t0, wind, t1, expected_ts, expected_q)
PRESETS = {
    "Custom input": None,
    "Case 1: Horiz. Cylinder | Lining 1 | 0 mph": ("hcyl", {"D": 8.0}, 1, [6.0], 0.8, 100.0, 0.0, 2000.0, 413, 991),
    "Case 2: Horiz. Cylinder | Lining 1 | 10 mph": ("hcyl", {"D": 8.0}, 1, [6.0], 0.8, 100.0, 10.0, 2000.0, 385, 1008),
    "Case 3: Horiz. Cylinder | Lining 1 | -5 °F": ("hcyl", {"D": 8.0}, 1, [6.0], 0.8, -5.0, 10.0, 2000.0, 330, 1042),
    "Case 4: Horiz. Cylinder | Lining 2 (variable k)": ("hcyl", {"D": 8.0}, 2, [6.0], 0.8, -5.0, 10.0, 2000.0, 78, 208),
    "Case 5: Vert. Cylinder | Lining 2 | 0 mph": ("vcyl", {"D": 6.0, "h": 30.0}, 2, [6.0], 0.8, 100.0, 0.0, 2000.0, 198, 201),
    "Case 6: Vert. Cylinder | Lining 3": ("vcyl", {"D": 6.0, "h": 30.0}, 3, [6.0, 4.0], 0.8, 100.0, 0.0, 2000.0, 388, 873),
    "Case 7: Vert. Cylinder | Lining 4": ("vcyl", {"D": 6.0, "h": 30.0}, 4, [6.0, 4.0, 1.5], 0.8, 100.0, 0.0, 2000.0, 272, 419),
    "Case 8: Vert. Cylinder | Lining 5 | 2500 °F": ("vcyl", {"D": 30.0, "h": 8.0}, 5, [6.0, 4.0, 1.5], 0.8, -5.0, 10.0, 2500.0, 235, 649),
    "Case 9: Vertical Wall (30x6 ft) | Lining 4": ("wall", {"h": 30.0, "l": 6.0}, 4, [6.0, 4.0, 1.5], 0.8, 100.0, 0.0, 2000.0, 283, 457),
    "Case 10: Vertical Wall (6x30 ft) | Lining 4": ("wall", {"h": 6.0, "l": 30.0}, 4, [6.0, 4.0, 1.5], 0.8, 100.0, 0.0, 2000.0, 281, 458),
    "Case 11: Vertical Wall | Lining 4 | -5 °F, 10 mph": ("wall", {"h": 6.0, "l": 30.0}, 4, [6.0, 4.0, 1.5], 0.8, -5.0, 10.0, 2000.0, 161, 475),
    "Case 12: Vertical Wall | Lining 5 | ε 0.15": ("wall", {"h": 6.0, "l": 30.0}, 5, [6.0, 4.0, 1.5], 0.15, -5.0, 10.0, 2500.0, 302, 651),
    "Case 13: Roof (6x30 ft) | Lining 4 | 0 mph": ("roof", {"l": 6.0, "w": 30.0}, 4, [6.0, 4.0, 1.5], 0.8, 100.0, 0.0, 2000.0, 267, 460),
    "Case 14: Roof (30x6 ft) | Lining 4 | 0 mph": ("roof", {"l": 30.0, "w": 6.0}, 4, [6.0, 4.0, 1.5], 0.8, 100.0, 0.0, 2000.0, 267, 460),
    "Case 15: Roof | Lining 4 | -5 °F, 10 mph": ("roof", {"l": 30.0, "w": 6.0}, 4, [6.0, 4.0, 1.5], 0.8, -5.0, 10.0, 2000.0, 158, 476),
    "Case 16: Roof | Lining 5 | 2500 °F, ε 0.15": ("roof", {"l": 30.0, "w": 6.0}, 5, [6.0, 4.0, 1.5], 0.15, -5.0, 10.0, 2500.0, 285, 654),
    "Case 17: Floor (6x30 ft) | Lining 4 | 0 mph": ("floor", {"l": 6.0, "w": 30.0}, 4, [6.0, 4.0, 1.5], 0.8, 100.0, 0.0, 2000.0, 317, 452),
    "Case 18: Floor (30x6 ft) | Lining 4 | 0 mph": ("floor", {"l": 30.0, "w": 6.0}, 4, [6.0, 4.0, 1.5], 0.8, 100.0, 0.0, 2000.0, 317, 452),
    "Case 19: Floor | Lining 4 | -5 °F, 10 mph": ("floor", {"l": 30.0, "w": 6.0}, 4, [6.0, 4.0, 1.5], 0.8, -5.0, 10.0, 2000.0, 167, 475),
    "Case 20: Floor | Lining 5 | 2500 °F, ε 0.15": ("floor", {"l": 30.0, "w": 6.0}, 5, [6.0, 4.0, 1.5], 0.15, -5.0, 10.0, 2500.0, 332, 647),
}


def k_poly(points):
    t, k = zip(*points)
    return np.poly1d(np.polyfit(t, k, min(3, len(points) - 1)))


def k_eff(poly, t1, t2):
    if abs(t2 - t1) < 1e-9:
        return max(float(poly(t1)), 1e-4)
    P = poly.integ()
    return max(float((P(t2) - P(t1)) / (t2 - t1)), 1e-4)


def air(tf):
    tk = max((tf - 32.0) / 1.8 + 273.15, 50.0)
    kf = 6.325e-6 * sqrt(tk) / (1.0 + 245.4 * (10.0 ** (-12.0 / tk)) / tk) * 241.77
    mu = 145.8 * tk * sqrt(tk) / (tk + 110.4) * 241.9e-7
    pr = max(0.7189 - tf * (1.6349e-4 - tf * (1.8106e-7 - 5.6617e-11 * tf)), 0.1)
    cp = max(0.24008 - tf * (1.2477e-6 - tf * (4.0489e-8 - 1.6088e-11 * tf)), 0.1)
    rho = 22.0493 / tk
    return kf, mu / rho, pr, cp, rho, 1.0 / (1.8 * tk), mu


def h_out(ts, t0, eps, wind, geom, d):
    Ts, To = max(ts + 459.67, 1.0), max(t0 + 459.67, 1.0)
    hr = SIGMA * eps * (Ts**3 + Ts**2 * To + Ts * To**2 + To**3)
    kf, nu, pr, cp, rho, beta, _ = air(0.5 * (ts + t0))
    dT = abs(Ts - To)
    V = max(wind, 0.0) * 5280.0
    Ra = lambda L: G * beta * rho * cp * dT * (L**3) / (nu * kf)

    if geom in ("wall", "vcyl"):
        Ln = d["h"]
        Nun = (0.825 + 0.387 * Ra(Ln)**(1/6) / (1 + (0.492 / pr)**(9/16))**(8/27))**2
    elif geom == "hcyl":
        Ln = d["D"]
        Nun = (0.60 + 0.387 * Ra(Ln)**(1/6) / (1 + (0.559 / pr)**(9/16))**(8/27))**2
    elif geom == "sphere":
        Ln = d["D"]
        Nun = 2.0 + 0.589 * Ra(Ln)**0.25 / (1 + (0.469 / pr)**(9/16))**(4/9)
    else:
        Ln = d["l"] * d["w"] / (2.0 * (d["l"] + d["w"]))
        r = Ra(Ln)
        heat_flow_up = (geom == "roof" and ts >= t0) or (geom == "floor" and ts < t0)
        Nun = (0.54 * r**0.25 if r < 1e7 else 0.15 * r**(1/3)) if heat_flow_up else 0.27 * r**0.25

    if geom in ("wall", "roof", "floor"):
        Lf = d["l"]
        Re = V * Lf / nu
        Nuf = (0.6744 * sqrt(Re) * pr**(1/3) / (1 + (0.0468 / pr)**(2/3))**0.25 if Re < 5e5
               else max((0.037 * Re**0.8 - 871.0) * pr**(1/3), 0.0))
    elif geom in ("hcyl", "vcyl"):
        Lf = d["D"]
        Re = V * Lf / nu
        Nuf = 0.3 + 0.62 * sqrt(Re) * pr**(1/3) / (1 + (0.4 / pr)**(2/3))**0.25 * (1 + (Re / 282000.0)**(5/8))**0.8
    else:
        Lf = d["D"]
        _, nu_inf, pr_inf, _, _, _, mu_inf = air(t0)
        _, _, _, _, _, _, mu_s = air(ts)
        Re_inf = V * Lf / nu_inf
        Nuf = 2.0 + (0.4 * sqrt(Re_inf) + 0.060 * Re_inf**(2/3)) * pr_inf**0.4 * (mu_inf / mu_s)**0.25

    if geom == "hcyl":
        Nu = 0.3 + ((Nuf - 0.3)**4 + (Nun - 0.3)**4)**0.25
        hc = Nu * kf / Lf
    elif geom == "sphere":
        Nu = 2.0 + (max(Nuf - 2.0, 0.0)**4 + max(Nun - 2.0, 0.0)**4)**0.25
        hc = Nu * kf / Lf
    else:
        j = 3.5 if geom in ("roof", "floor") else 3.0
        hc = ((Nuf * kf / Lf)**j + (Nun * kf / Ln)**j)**(1.0 / j)

    return hr + hc, hr, hc


def solve_api985(geom, d, lining_id, thick, eps, t0, wind, t1):
    _, layers = LININGS[lining_id]
    polys = [k_poly(p) for _, _, p in layers]
    n = len(polys)

    if geom in ("hcyl", "vcyl", "sphere") and sum(thick) >= d["D"] * 6.0:
        raise ValueError(f"Total refractory thickness ({sum(thick)} in.) exceeds vessel inner radius ({d['D']*6.0} in.).")

    x = np.concatenate(([0.0], np.cumsum(thick, dtype=float)))
    if geom in ("hcyl", "vcyl", "sphere"):
        x = d["D"] * 6.0 - x[-1] + x

    ts_init = t0 + 0.1 * (t1 - t0)
    t = np.concatenate(([t1], t1 + (ts_init - t1) * (x[1:] - x[0]) / (x[-1] - x[0])))

    alpha, prev_err = 1.0, float("inf")
    for it in range(500):
        ke = [k_eff(polys[m], t[m + 1], t[m]) for m in range(n)]
        if geom in ("hcyl", "vcyl"):
            R = [x[-1] * log(x[m + 1] / x[m]) / ke[m] for m in range(n)]
        elif geom == "sphere":
            R = [(x[-1]**2) * (x[m + 1] - x[m]) / (ke[m] * x[m] * x[m + 1]) for m in range(n)]
        else:
            R = [(x[m + 1] - x[m]) / ke[m] for m in range(n)]

        h_tot, hr, hc = h_out(t[-1], t0, eps, wind, geom, d)
        Ro = 1.0 / h_tot
        q = (t1 - t0) / (sum(R) + Ro)

        new = [t1]
        for m in range(n):
            new.append(new[-1] - q * R[m])
        new = np.array(new, dtype=float)

        err = float(np.max(np.abs(new - t)))
        if err < 0.1:
            t = new
            ke = [k_eff(polys[m], t[m + 1], t[m]) for m in range(n)]
            return t, q, ke, R, h_tot, hr, hc, it + 1

        if err > prev_err and it > 3:
            alpha = max(0.25, alpha * 0.7)
        prev_err = err
        t = (1.0 - alpha) * t + alpha * new

    raise RuntimeError("Solver did not converge within 500 iterations.")


# ==========================================
# 2. UI
# ==========================================
st.set_page_config(page_title="API RP 985 Calculator", page_icon="🔥", layout="centered",
                   initial_sidebar_state="collapsed")

PALETTE = ["#e4572e", "#f3a712", "#5b9bd5", "#7a8b99"]  # hot face -> backup

st.markdown(
    """
<style>
.block-container {padding: 1rem 0.9rem 3rem; max-width: 760px;}
h1 {font-size: 1.5rem !important; margin-bottom: 0 !important;}
.sub {opacity: .65; font-size: .85rem; margin-bottom: .8rem;}
.hero {border-radius: 18px; padding: 18px 16px; margin: 8px 0 12px; color: #fff;}
.hero .lbl {font-size: .8rem; opacity: .9; letter-spacing: .04em; text-transform: uppercase;}
.hero .big {font-size: 3rem; font-weight: 800; line-height: 1.1;}
.hero .note {font-size: .85rem; opacity: .95; margin-top: 2px;}
.grid {display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px;}
.kpi {border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: 10px 12px;}
.kpi .l {font-size: .72rem; opacity: .65; text-transform: uppercase; letter-spacing: .03em;}
.kpi .v {font-size: 1.25rem; font-weight: 700;}
.kpi .u {font-size: .72rem; opacity: .6;}
.sec {font-size: 1.05rem; font-weight: 700; margin: 18px 0 6px;}
.lay {display: flex; gap: 10px; border: 1px solid rgba(128,128,128,.28); border-radius: 14px;
      padding: 10px 12px; margin-bottom: 8px;}
.lay .bar {width: 6px; border-radius: 4px; flex: none;}
.lay .nm {font-weight: 700; font-size: .95rem;}
.lay .tp {font-size: .8rem; opacity: .75;}
.lay .tm {font-size: .78rem; opacity: .6; margin-top: 2px;}
.stack {display: flex; height: 26px; border-radius: 8px; overflow: hidden; margin: 6px 0;}
.leg {display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: .78rem; opacity: .85;}
.leg i {display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px;}
.chk {font-size: .8rem; opacity: .7; margin-top: 10px; text-align: center;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🔥 Refractory Shell Temperature")
st.markdown('<div class="sub">API RP 985 steady-state heat loss calculator</div>', unsafe_allow_html=True)

GEOM_LABELS = {
    "hcyl": "Horizontal cylinder",
    "vcyl": "Vertical cylinder",
    "wall": "Vertical wall",
    "roof": "Roof (heat flows up)",
    "floor": "Floor (heat flows down)",
    "sphere": "Sphere",
}
LABEL_TO_GEOM = {v: k for k, v in GEOM_LABELS.items()}

# ---------- Inputs (in main page so they work on mobile) ----------
with st.expander("⚙️ Inputs", expanded=True):
    preset_name = st.selectbox("Start from a benchmark case", list(PRESETS.keys()))
    preset = PRESETS[preset_name]
    K = preset_name  # widget keys change with preset so values reload

    tab_g, tab_l, tab_c = st.tabs(["📐 Shape", "🧱 Lining", "🌡️ Conditions"])

    with tab_g:
        default_geom = GEOM_LABELS[preset[0]] if preset else "Horizontal cylinder"
        geom_label = st.selectbox("Equipment type", list(GEOM_LABELS.values()),
                                  index=list(GEOM_LABELS.values()).index(default_geom), key=f"geom_{K}")
        geom = LABEL_TO_GEOM[geom_label]
        d = {}
        if geom in ("hcyl", "vcyl", "sphere"):
            d["D"] = st.number_input("Shell inside diameter (ft)", min_value=0.5,
                                     value=float(preset[1].get("D", 8.0)) if preset else 8.0, step=0.5, key=f"D_{K}")
        if geom in ("vcyl", "wall"):
            d["h"] = st.number_input("Height (ft)", min_value=0.5,
                                     value=float(preset[1].get("h", 30.0)) if preset else 30.0, step=1.0, key=f"h_{K}")
        if geom in ("wall", "roof", "floor"):
            d["l"] = st.number_input("Length in wind direction (ft)", min_value=0.5,
                                     value=float(preset[1].get("l", 30.0)) if preset else 30.0, step=1.0, key=f"l_{K}")
        if geom in ("roof", "floor"):
            d["w"] = st.number_input("Width (ft)", min_value=0.5,
                                     value=float(preset[1].get("w", 6.0)) if preset else 6.0, step=1.0, key=f"w_{K}")

    with tab_l:
        def_lin = preset[2] if preset else 4
        lining_id = st.selectbox("Lining system", list(LININGS.keys()),
                                 format_func=lambda k: LININGS[k][0],
                                 index=list(LININGS.keys()).index(def_lin), key=f"lin_{K}")
        _, layers_info = LININGS[lining_id]
        thick = []
        for idx, (lname, def_th, _) in enumerate(layers_info):
            pth = float(preset[3][idx]) if (preset and preset[2] == lining_id and idx < len(preset[3])) else float(def_th)
            thick.append(st.number_input(f"{idx+1}. {lname} thickness (in.)", min_value=0.1, value=pth,
                                         step=0.5, key=f"th_{K}_{lining_id}_{idx}"))

    with tab_c:
        t1 = st.number_input("Hot face temperature (°F)", -200.0, 4000.0,
                             float(preset[7]) if preset else 2000.0, 25.0, key=f"t1_{K}")
        t0 = st.number_input("Ambient temperature (°F)", -100.0, 500.0,
                             float(preset[5]) if preset else 100.0, 5.0, key=f"t0_{K}")
        wind = st.number_input("Wind speed (mph)", 0.0, 150.0,
                               float(preset[6]) if preset else 0.0, 1.0, key=f"wind_{K}")
        eps = st.slider("Shell emissivity (ε)", 0.05, 1.00,
                        float(preset[4]) if preset else 0.80, 0.05, key=f"eps_{K}")

# ---------- Solve ----------
try:
    t_prof, q_flux, ke_list, R_list, h_tot, hr, hc, iters = solve_api985(geom, d, lining_id, thick, eps, t0, wind, t1)
except Exception as e:
    st.error(f"⚠️ {e}")
    st.stop()

ts = float(t_prof[-1])
if ts < 200:
    grad, status = "linear-gradient(135deg,#2e9e6b,#1f7a52)", "Cool shell"
elif ts < 300:
    grad, status = "linear-gradient(135deg,#e0a21b,#c27d0e)", "Warm shell"
else:
    grad, status = "linear-gradient(135deg,#e4572e,#b83a17)", "Hot shell"

# ---------- Hero ----------
st.markdown(
    f"""
<div class="hero" style="background:{grad}">
  <div class="lbl">Predicted shell temperature</div>
  <div class="big">{ts:.0f} °F</div>
  <div class="note">{status} · {ts - t0:+.0f} °F vs ambient</div>
</div>
""",
    unsafe_allow_html=True,
)

total_th = sum(thick)
if geom in ("hcyl", "vcyl"):
    third = (f"{pi * d['D'] * q_flux:,.0f}", "Btu/(h·ft) of length", "Heat loss per length")
else:
    third = (f"{total_th:.1f}", "inches", "Lining thickness")

st.markdown(
    f"""
<div class="grid">
  <div class="kpi"><div class="l">Heat loss</div><div class="v">{q_flux:,.0f}</div><div class="u">Btu/(h·ft²)</div></div>
  <div class="kpi"><div class="l">{third[2]}</div><div class="v">{third[0]}</div><div class="u">{third[1]}</div></div>
  <div class="kpi"><div class="l">Outer film h</div><div class="v">{h_tot:.2f}</div><div class="u">Btu/(h·ft²·°F)</div></div>
  <div class="kpi"><div class="l">Radiation / convection</div><div class="v">{hr:.1f} / {hc:.1f}</div><div class="u">Btu/(h·ft²·°F)</div></div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- Temperature profile chart ----------
st.markdown('<div class="sec">Temperature through the lining</div>', unsafe_allow_html=True)

edges = np.concatenate(([0.0], np.cumsum(thick)))
names = [n for n, _, _ in layers_info]
bands = pd.DataFrame({
    "Layer": names,
    "x0": edges[:-1],
    "x1": edges[1:],
    "y0": [min(t0, ts) - 0.05 * (t1 - t0)] * len(names),
    "y1": [t1 + 0.05 * (t1 - t0)] * len(names),
})
color_scale = alt.Scale(domain=names, range=PALETTE[:len(names)])
bg = alt.Chart(bands).mark_rect(opacity=0.22).encode(
    x=alt.X("x0:Q", title="Distance from hot face (in.)"),
    x2="x1:Q",
    y=alt.Y("y0:Q", title="Temperature (°F)"),
    y2="y1:Q",
    color=alt.Color("Layer:N", scale=color_scale, legend=alt.Legend(orient="bottom", title=None)),
)
line_df = pd.DataFrame({"x": edges, "T": t_prof})
line = alt.Chart(line_df).mark_line(strokeWidth=3, color="#d62728", point=alt.OverlayMarkDef(size=70, color="#d62728")).encode(
    x="x:Q", y="T:Q", tooltip=[alt.Tooltip("x:Q", title="Depth (in.)", format=".1f"),
                                alt.Tooltip("T:Q", title="Temp (°F)", format=".0f")])
labels = alt.Chart(line_df).mark_text(dy=-12, fontWeight="bold", fontSize=12).encode(
    x="x:Q", y="T:Q", text=alt.Text("T:Q", format=".0f"))
chart = (bg + line + labels).properties(height=300).configure_axis(grid=False)
st.altair_chart(chart, width="stretch")

# ---------- Layer cards ----------
st.markdown('<div class="sec">Layers</div>', unsafe_allow_html=True)
cards = ""
for i, nm in enumerate(names):
    cards += f"""
<div class="lay">
  <div class="bar" style="background:{PALETTE[i]}"></div>
  <div style="flex:1">
    <div class="nm">{nm} · {thick[i]:g} in.</div>
    <div class="tp">{t_prof[i]:.0f} °F → {t_prof[i+1]:.0f} °F &nbsp;(drop {t_prof[i]-t_prof[i+1]:.0f} °F)</div>
    <div class="tm">k<sub>eff</sub> {ke_list[i]:.2f} Btu·in/(h·ft²·°F) · R {R_list[i]:.3f}</div>
  </div>
</div>"""
st.markdown(cards, unsafe_allow_html=True)

# ---------- Where the temperature drop happens ----------
st.markdown('<div class="sec">Where the temperature drop happens</div>', unsafe_allow_html=True)
drops = [float(t_prof[i] - t_prof[i + 1]) for i in range(len(names))] + [max(ts - t0, 0.0)]
dn = names + ["Outside air film"]
dc = PALETTE[:len(names)] + ["#9aa5b1"]
tot = sum(drops) or 1.0
seg = "".join(
    f'<div style="width:{v / tot * 100:.2f}%;background:{c}" title="{n}: {v:.0f} °F"></div>'
    for v, c, n in zip(drops, dc, dn)
)
leg = "".join(
    f'<span><i style="background:{c}"></i>{n} {v / tot * 100:.0f}%</span>' for v, c, n in zip(drops, dc, dn)
)
st.markdown(f'<div class="stack">{seg}</div><div class="leg">{leg}</div>', unsafe_allow_html=True)

# ---------- Benchmark check ----------
if preset:
    ok = abs(ts - preset[8]) <= 2 and abs(q_flux - preset[9]) <= 3
    icon = "✅" if ok else "⚠️"
    st.markdown(
        f'<div class="chk">{icon} Table A.8 target: {preset[8]} °F · {preset[9]} Btu/(h·ft²) '
        f'&nbsp;|&nbsp; calculated: {ts:.0f} °F · {q_flux:.0f}</div>',
        unsafe_allow_html=True,
    )
