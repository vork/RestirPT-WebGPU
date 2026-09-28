"""Static HTML report generator for compare.py (template.html + render_index)."""
from __future__ import annotations

import html
import json
from pathlib import Path
from string import Template

_TEMPLATE = Path(__file__).with_name("template.html")


def _e(x) -> str:
    return html.escape(str(x))


def _num(x, fmt: str = ".4g") -> str:
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, float)):
        if abs(x) >= 1e300:
            return "inf" if x > 0 else "-inf"
        return format(x, fmt)
    return _e(x)


def _pct(x) -> str:
    return "–" if x is None else ("inf" if abs(x) >= 1e300 else f"{100 * x:.3f}%")


def _badge(ok: bool | None, label: str | None = None) -> str:
    if ok is None:
        return '<span class="badge na">○ n/a</span>'
    cls, icon = ("pass", "✓") if ok else ("fail", "✗")
    return f'<span class="badge {cls}">{icon} {_e(label or ("pass" if ok else "fail"))}</span>'


def _table(head: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def _checks(report: dict) -> str:
    rows = []
    for c in report.get("checks", []):
        applicable = c.get("applicable", True)
        rows.append([_e(c["name"]), _e(c["channel"]), _badge(c["passed"] if applicable else None),
                     "gating" if c.get("gating", True) else "info", _num(c.get("value")), _num(c.get("threshold"))])
    return _table(["check", "channel", "result", "role", "value", "threshold"], rows)


def _channels(report: dict) -> str:
    chans = report.get("channels", {})
    if not chans:
        return ""
    rows = []
    for ch, v in chans.items():
        g, t = v["global_"], v["tiles"]
        rows.append([_e(ch), _pct(g["rel"]), _pct(g["se_rel"]), _pct(g["bound_rel"]), _pct(g["delta_margin"]),
                     _pct(g["mdb"]), _num(g["t"], ".3f"), _num(g["nu"], ".1f"), _badge(g["passed"])])
    out = "<h3>Global mean (TOST, α per side)</h3>" + _table(
        ["ch", "Δ/R̄", "SE/R̄", "|Δ|+t·SE", "δ", "MDB", "t", "ν", "TOST"], rows)
    rows = []
    for ch, v in chans.items():
        t = v["tiles"]
        c2, mt, ka = t["chi2_red"], t["mean_t"], t["ks_ad"]
        rows.append([_e(ch), str(t["m"]), str(t["tost_failed"]), str(t["dark"]), str(t["zero_var"]),
                     _num(t["max_ratio"], ".3f"), _pct(t["mdb_max"]), str(t["sidak_rejected"]), str(t["bh_rejected"]),
                     _num(c2.get("stat"), ".3f") + " / " + _num(c2.get("bound"), ".3f"),
                     _num(mt.get("z"), ".2f"), _num(ka.get("p_ks"), ".3g") + " / " + _num(ka.get("p_ad"), ".3g")])
    out += "<h3>Tiles</h3>" + _table(
        ["ch", "m", "TOST fail", "dark", "zero-var", "max bound/margin", "MDB max", "Šidák rej", "BH rej",
         "χ²_red / bound", "mean-t z", "p KS / AD"], rows)
    mrows = []
    for ch, v in chans.items():
        for m in v.get("masks", []):
            mrows.append([_e(ch), _e(m["name"]), _pct(m["rel"]), _pct(m["se_rel"]), _num(m["ratio"], ".3f"),
                          _pct(m["mdb"]), _badge(m["passed"])])
    if mrows:
        out += "<h3>Mask regions</h3>" + _table(["ch", "mask", "Δ/R̄", "SE/R̄", "bound/margin", "MDB", "TOST"], mrows)
    return out


def _kv(d: dict) -> str:
    rows = [[_e(k), f"<code>{_e(json.dumps(v) if isinstance(v, (dict, list)) else v)}</code>"] for k, v in d.items()]
    return _table(["key", "value"], rows)


def render_index(report: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    status = report.get("status", "fail")
    ok = status in ("pass", "pass_on_rerun")
    images = "".join(
        f'<figure><img src="{_e(f)}" alt="{_e(k)}" loading="lazy"><figcaption>{_e(k)}</figcaption></figure>'
        for k, f in report.get("images", {}).items())
    sections = []
    if report.get("checks"):
        sections.append("<section><h2>Gate checks</h2>" + _checks(report) + "</section>")
    ch = _channels(report)
    if ch:
        sections.append("<section><h2>Per-channel statistics</h2>" + ch + "</section>")
    for key, title in (("metrics", "Metrics (Y, converged means)"), ("sizing", "Sizing"),
                       ("curve", "Convergence curve"), ("calibration", "Calibration (Gate 1)"),
                       ("decision", "Suite decision"), ("provenance", "Provenance"), ("inputs", "Inputs"),
                       ("test", "Test declaration")):
        if report.get(key):
            sections.append(f"<section><h2>{title}</h2>{_kv(report[key])}</section>")
    if images:
        sections.append(f'<section><h2>Figures</h2><div class="figs">{images}</div></section>')
    page = Template(_TEMPLATE.read_text()).substitute(
        title=_e(report.get("test", {}).get("name") or "compare.py report"),
        mode=_e(report.get("mode", "compare")),
        status_badge=_badge(ok, status),
        subtitle=_e(f"stage {report.get('test', {}).get('stage', '?')} · created {report.get('created', '')}"),
        failed=_e(", ".join(report.get("failed_checks", [])) or "none"),
        sections="\n".join(sections),
    )
    path = out / "index.html"
    path.write_text(page)
    return path
