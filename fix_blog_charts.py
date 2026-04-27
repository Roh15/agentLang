import json
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Color palette from blog.html ──────────────────────────────────────────────
BG        = "#0f1117"
SURFACE   = "#1a1d27"
BORDER    = "#2e3350"
TEXT      = "#e2e8f0"
MUTED     = "#8892b0"
ACCENT    = "#64ffda"
RED       = "#ff6b6b"
YELLOW    = "#ffd166"
GREEN     = "#06d6a0"
BLUE      = "#4facfe"

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=SURFACE,
    font=dict(family="Inter, -apple-system, sans-serif", color=TEXT, size=13),
    margin=dict(t=30, r=24, b=50, l=70),
)

AXIS_STYLE = dict(
    gridcolor=BORDER,
    zerolinecolor=BORDER,
    linecolor=BORDER,
    tickcolor=MUTED,
    tickfont=dict(color=MUTED, size=12),
)

LEGEND_STYLE = dict(
    bgcolor="rgba(26,29,39,0.85)",
    bordercolor=BORDER,
    borderwidth=1,
    font=dict(color=TEXT, size=12),
)

# ── Load data ─────────────────────────────────────────────────────────────────
with open("results/results_20260426_145210.json") as f:
    exp001 = json.load(f)

with open("results/exp003_scaling_20260426_151700.json") as f:
    exp003 = json.load(f)

with open("results/exp004_7b_20260426_154730.json") as f:
    exp004 = json.load(f)

with open("results/exp006_prompt_sweep_20260426_162720.json") as f:
    exp006 = json.load(f)


# ── Figure 1: EXP-001 prefill timing comparison ───────────────────────────────
def make_fig1():
    std = exp001["experiment_1"]["standard_prefill_ms"]
    kv  = exp001["experiment_1"]["kv_prefill_ms"]

    fig = go.Figure()
    for name, vals, color, fillcolor in [
        ("Standard Pipeline", std, RED,    "rgba(255,107,107,0.2)"),
        ("KV Cache Pipeline", kv,  ACCENT, "rgba(100,255,218,0.2)"),
    ]:
        fig.add_trace(go.Box(
            y=vals,
            name=name,
            marker_color=color,
            line_color=color,
            fillcolor=fillcolor,
            boxpoints="all",
            jitter=0.35,
            pointpos=0,
            marker=dict(size=5, opacity=0.7, color=color),
            boxmean=True,
        ))

    fig.update_layout(
        **BASE_LAYOUT,
        height=340,
        yaxis=dict(title="Prefill Time (ms)", **AXIS_STYLE),
        xaxis=dict(**AXIS_STYLE),
        legend=dict(x=0.55, y=0.98, **LEGEND_STYLE),
        bargap=0.3,
    )
    return fig


# ── Figure 2: EXP-001 multi-hop prefill scaling ───────────────────────────────
def make_fig2():
    hops = exp001["experiment_3"]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["2-hop", "3-hop", "4-hop"],
        shared_yaxes=True,
    )

    showlegend = True
    for col, hop_n in enumerate([2, 3, 4], start=1):
        data = hops[str(hop_n)]
        agents = data["agent_keys"]
        std_vals = data["standard_prefill_ms"]
        kv_vals  = data["kv_prefill_ms"]

        fig.add_trace(go.Bar(
            name="Standard Pipeline",
            x=agents,
            y=std_vals,
            marker_color=RED,
            marker_line_color="rgba(0,0,0,0)",
            showlegend=showlegend,
            legendgroup="std",
        ), row=1, col=col)

        fig.add_trace(go.Bar(
            name="KV Cache Pipeline",
            x=agents,
            y=kv_vals,
            marker_color=ACCENT,
            marker_line_color="rgba(0,0,0,0)",
            showlegend=showlegend,
            legendgroup="kv",
        ), row=1, col=col)

        showlegend = False

    fig.update_layout(
        **BASE_LAYOUT,
        height=320,
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
        legend=dict(x=0.68, y=0.98, **LEGEND_STYLE),
    )
    for i in range(1, 4):
        fig.update_xaxes(**AXIS_STYLE, row=1, col=i)
        fig.update_yaxes(**AXIS_STYLE, row=1, col=i)
    fig.update_yaxes(title_text="Prefill Time (ms)", row=1, col=1)

    # subplot title color
    for ann in fig.layout.annotations:
        ann.font.color = MUTED
        ann.font.size  = 12

    return fig


# ── Figure 3: EXP-003 scaling curve ──────────────────────────────────────────
def make_fig3():
    sweep = exp003["standard_sweep"]
    xs     = [s["actual_len"] for s in sweep]
    means  = [s["mean_ms"]    for s in sweep]
    stds   = [s["std_ms"]     for s in sweep]

    kv_mean = exp003["kv_prefix_baseline"]["mean_ms"]
    kv_std  = exp003["kv_prefix_baseline"]["std_ms"]

    fig = go.Figure()

    # error band via filled scatter
    fig.add_trace(go.Scatter(
        x=xs + xs[::-1],
        y=[m + s for m, s in zip(means, stds)] + [m - s for m, s in zip(means, stds)][::-1],
        fill="toself",
        fillcolor=f"rgba(255,107,107,0.15)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=xs, y=means,
        mode="lines+markers",
        name="Standard Pipeline",
        line=dict(color=RED, width=2.5),
        marker=dict(color=RED, size=7, symbol="circle"),
    ))

    # KV baseline band
    fig.add_hrect(
        y0=kv_mean - kv_std, y1=kv_mean + kv_std,
        fillcolor=f"rgba(100,255,218,0.1)",
        line_width=0,
        layer="below",
    )
    fig.add_hline(
        y=kv_mean,
        line=dict(color=ACCENT, width=2, dash="dash"),
        annotation_text=f"KV baseline {kv_mean:.1f} ms",
        annotation_position="top right",
        annotation_font_color=ACCENT,
        annotation_font_size=12,
    )

    fig.update_layout(
        **BASE_LAYOUT,
        height=340,
        xaxis=dict(title="Input Tokens", **AXIS_STYLE),
        yaxis=dict(title="Prefill Time (ms)", **AXIS_STYLE),
        legend=dict(x=0.04, y=0.96, **LEGEND_STYLE),
    )
    return fig


# ── Figure 4: EXP-004 7B comparison ──────────────────────────────────────────
def make_fig4():
    std_3b = exp001["experiment_1"]["standard_prefill_ms"]
    kv_3b  = exp001["experiment_1"]["kv_prefill_ms"]
    std_7b = exp004["part1_prefill"]["standard_prefill_ms"]
    kv_7b  = exp004["part1_prefill"]["kv_prefill_ms"]

    import statistics
    data = {
        "3B Standard": (statistics.mean(std_3b), statistics.stdev(std_3b), RED),
        "3B KV Cache": (statistics.mean(kv_3b),  statistics.stdev(kv_3b),  ACCENT),
        "7B Standard": (statistics.mean(std_7b), statistics.stdev(std_7b), "#ff9999"),
        "7B KV Cache": (statistics.mean(kv_7b),  statistics.stdev(kv_7b),  GREEN),
    }

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["3B Model", "7B Model"],
        shared_yaxes=True,
    )

    for col, (model, pairs) in enumerate([
        ("3B", [("Standard Pipeline", statistics.mean(std_3b), statistics.stdev(std_3b), RED),
                ("KV Cache Pipeline", statistics.mean(kv_3b),  statistics.stdev(kv_3b),  ACCENT)]),
        ("7B", [("Standard Pipeline", statistics.mean(std_7b), statistics.stdev(std_7b), RED),
                ("KV Cache Pipeline", statistics.mean(kv_7b),  statistics.stdev(kv_7b),  ACCENT)]),
    ], start=1):
        for name, mean, std, color in pairs:
            fig.add_trace(go.Bar(
                name=name,
                x=[name.replace(" Pipeline", "")],
                y=[mean],
                error_y=dict(type="data", array=[std], visible=True, color=color, thickness=1.5, width=6),
                marker_color=color,
                marker_line_color="rgba(0,0,0,0)",
                showlegend=(col == 1),
                legendgroup=name,
                text=[f"{mean:.1f} ms"],
                textposition="outside",
                textfont=dict(color=TEXT, size=12),
            ), row=1, col=col)

    fig.update_layout(
        **BASE_LAYOUT,
        height=340,
        barmode="group",
        bargap=0.35,
        legend=dict(x=0.6, y=0.98, **LEGEND_STYLE),
    )
    for i in range(1, 3):
        fig.update_xaxes(**AXIS_STYLE, row=1, col=i)
        fig.update_yaxes(**AXIS_STYLE, row=1, col=i)
    fig.update_yaxes(title_text="Prefill Time (ms)", row=1, col=1)
    for ann in fig.layout.annotations:
        ann.font.color = MUTED
        ann.font.size  = 12

    return fig


# ── Figure 5: EXP-006 heatmap ────────────────────────────────────────────────
def make_fig5():
    grid    = exp006["speedup_grid"]
    prior   = [str(x) for x in exp006["prior_context_lengths"]]
    sysprom = [str(x) for x in exp006["system_prompt_lengths"]]

    # annotate each cell
    text = [[f"{v:.1f}×" for v in row] for row in grid]

    colorscale = [
        [0.0,  "#1a1d27"],
        [0.15, "#1f2d3d"],
        [0.35, "#16425b"],
        [0.55, "#0d6e6e"],
        [0.75, "#06a87a"],
        [1.0,  ACCENT],
    ]

    fig = go.Figure(go.Heatmap(
        z=grid,
        x=[f"{s} tok" for s in sysprom],
        y=[f"{p} tok" for p in prior],
        text=text,
        texttemplate="%{text}",
        textfont=dict(color=TEXT, size=13, family="Inter, sans-serif"),
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            title=dict(text="Speedup", font=dict(color=MUTED, size=12)),
            tickfont=dict(color=MUTED, size=11),
            ticksuffix="×",
            outlinewidth=0,
            bgcolor="rgba(0,0,0,0)",
            thickness=14,
        ),
        hoverongaps=False,
        hovertemplate="Prior: %{y}<br>Sys prompt: %{x}<br>Speedup: %{text}<extra></extra>",
    ))

    fig.update_layout(
        **BASE_LAYOUT,
        height=320,
        xaxis=dict(title="System Prompt Length", **AXIS_STYLE),
        yaxis=dict(title="Prior Context Length", **AXIS_STYLE),
    )
    return fig


# ── Generate chart HTML fragments ─────────────────────────────────────────────
CHART_FUNCS = {
    "EXP-001: Prefill timing comparison":    make_fig1,
    "EXP-001: Multi-hop prefill scaling":    make_fig2,
    "EXP-003: Prefill scaling with sequence length": make_fig3,
    "EXP-004: 7B model results":             make_fig4,
    "EXP-006: System prompt sensitivity heatmap": make_fig5,
}

chart_html = {}
for alt, fn in CHART_FUNCS.items():
    fig = fn()
    chart_html[alt] = fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True, "displayModeBar": False},
    )

# ── Patch blog.html ───────────────────────────────────────────────────────────
with open("blog.html") as f:
    html = f.read()

# 1. Add Plotly CDN before </head>
plotly_cdn = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>\n'
html = html.replace("</head>", plotly_cdn + "</head>", 1)

# 2. Replace each base64 <img> with the corresponding Plotly chart
def replace_img(match):
    alt = re.search(r'alt="([^"]+)"', match.group(0))
    if alt and alt.group(1) in chart_html:
        return chart_html[alt.group(1)]
    return match.group(0)

html = re.sub(
    r'<img\s+src="data:image/png;base64,[^"]*"[^>]*>',
    replace_img,
    html,
)

with open("blog.html", "w") as f:
    f.write(html)

print("Done. Charts replaced:")
for alt in CHART_FUNCS:
    print(f"  ✓ {alt}")
