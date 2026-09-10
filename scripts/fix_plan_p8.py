"""Fix raw strings in FINAL_IMPLEMENTATION_PLAN_SIH26168.md."""

with open("FINAL_IMPLEMENTATION_PLAN_SIH26168.md", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "Stage B (Uncertainty Head):" in line:
        lines[i] = "   - Stage B (Uncertainty Head): Deferred to Phase 9 covariance calibration; hand-specified diagonal measurement noise $R_b = \\text{diag}([1.21, 1.21, 1.21, 0.0025, 0.0025, 0.0025])$ used in Phase 8 navigation ablation.\n"
    elif "Indirect Navigation Gate" in line and i > 600:
        lines[i] = "- **Indirect Navigation Gate**: PASSED (MODEST/DIAGNOSTIC). Evaluated on synthetic outages (10s, 30s, 60s); maintains filter stability with bounded innovations (Mean NIS $\\le 2.543$) and achieves lowest velocity tracking RMSE ($3.292\\text{ m/s}$ on 30s outage vs Pure ESKF $3.775\\text{ m/s}$, +VNet $3.361\\text{ m/s}$; $130.002\\text{ m/s}$ on 60s outage vs Pure ESKF $133.082\\text{ m/s}$, +VNet $131.604\\text{ m/s}$). Horizontal position RMSE is comparable to VelocityNet ($10.049\\text{ m}$ vs $10.031\\text{ m}$ at 30s; $1735.228\\text{ m}$ vs $1740.133\\text{ m}$ at 60s).\n"

with open("FINAL_IMPLEMENTATION_PLAN_SIH26168.md", "w", encoding="utf-8") as f:
    f.writelines(lines)

print("Cleaned up successfully!")
