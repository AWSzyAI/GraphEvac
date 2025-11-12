import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def _ensure_dir(p: str):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _fmt_hms(seconds: float) -> str:
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


def plot_paths(
    coords: Dict[str, Tuple[float, float]],
    rooms: List[str],
    start_node: Dict[str, str],
    result: dict,
    savepath: str = "out/topo_paths.png",
):
    _ensure_dir(savepath)
    import numpy as np

    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    # Heatmap based on first-clear times
    first = result.get("first_clear", {})
    by_room = first.get("by_room", {})
    times = [rec.get("time") for rec in by_room.values() if rec.get("time") is not None]
    t_min = min(times) if times else 0.0
    t_max = max(times) if times else 1.0
    cmap = plt.cm.inferno
    rx = np.array([coords[r][0] for r in rooms])
    ry = np.array([coords[r][1] for r in rooms])
    room_scores = []
    for r in rooms:
        t = by_room.get(r, {}).get("time")
        if t is None or t_max == t_min:
            room_scores.append(0.0)
        else:
            room_scores.append((t - t_min) / (t_max - t_min))
    room_scores = np.array(room_scores)
    rooms_sc = ax.scatter(
        rx,
        ry,
        marker="s",
        s=120,
        c=room_scores,
        cmap=cmap,
        edgecolors="#555555",
        linewidths=0.8,
    )
    cbar = fig.colorbar(rooms_sc, ax=ax, pad=0.03)
    cbar.set_label("First-clear time (s)")
    for r in rooms:
        ax.text(coords[r][0], coords[r][1] + 0.5, r, fontsize=8, color="#111", ha="center")

    # Draw exits (prefer meta exits if present)
    exits = set(result.get("meta", {}).get("exits", list(start_node.values())))
    ex = [coords[e][0] for e in exits]
    ey = [coords[e][1] for e in exits]
    ax.scatter(ex, ey, marker="^", s=80, c="#2ca02c", label="Exits")
    for e in exits:
        ax.text(
            coords[e][0] + 0.8,
            coords[e][1] + 0.3,
            e,
            fontsize=8,
            color="#2ca02c",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.6, edgecolor="none"),
        )

    # First-clear labels offset
    for r, rec in by_room.items():
        t = rec.get("time")
        by = rec.get("by")
        if t is None:
            continue
        txt = f"{_fmt_hms(t)}\nby {by}"
        x, y = coords[r]
        ax.text(
            x + 0.8,
            y + 0.4,
            txt,
            fontsize=8,
            color="#1f77b4",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
        )

    # Add total makespan + full order at lower left
    order = first.get("order", [])
    makespan = result.get("T_total")
    if makespan is not None:
        order_str = " -> ".join(order) if order else "N/A"
        x_min = min(np.concatenate([rx, ex])) if len(rx) + len(ex) else 0
        y_min = min(np.concatenate([ry, ey])) if len(ry) + len(ey) else 0
        text = f"Makespan: {_fmt_hms(makespan)}\nOrder: {order_str}"
        ax.text(
            x_min,
            y_min - 1.0,
            text,
            fontsize=8,
            color="#111",
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#666"),
        )

    # Draw paths per responder
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    for idx, (k, info) in enumerate(result.get("responders", {}).items()):
        c = colors[idx % len(colors)]
        path = info.get("path", [])
        for (u, v) in path:
            if u.startswith("G_") or v.startswith("G_"):
                # draw to nearest exit as a dashed line
                if v.startswith("G_"):
                    # from room u to nearest exit
                    room = u
                else:
                    room = v
                # find nearest exit
                e_best = None
                d_best = 1e18
                for e in exits:
                    dx = coords[room][0] - coords[e][0]
                    dy = coords[room][1] - coords[e][1]
                    d = (dx * dx + dy * dy) ** 0.5
                    if d < d_best:
                        d_best = d
                        e_best = e
                ax.plot(
                    [coords[room][0], coords[e_best][0]],
                    [coords[room][1], coords[e_best][1]],
                    linestyle="--",
                    color=c,
                    alpha=0.8,
                    label=f"{k} return" if u == path[0][0] and v == path[0][1] else None,
                )
            else:
                ax.annotate(
                    "",
                    xy=(coords[v][0], coords[v][1]),
                    xytext=(coords[u][0], coords[u][1]),
                    arrowprops=dict(arrowstyle="->", color=c, lw=2, alpha=0.8),
                )
        ax.scatter([], [], color=c, label=f"Path {k}")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Responder Paths and First-Clear Times")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout(pad=1.4)
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def plot_gantt(
    coords: Dict[str, Tuple[float, float]],
    rooms: List[str],
    start_node: Dict[str, str],
    result: dict,
    savepath: str = "out/gantt.png",
):
    _ensure_dir(savepath)
    walk_speed = result.get("meta", {}).get("walk_speed_used", 1.0)
    sweep_time = result.get("sweep_time", {})
    redundancy_mode = result.get("meta", {}).get("redundancy_mode", "assignment")

    # Build segments per responder
    segs = {}
    for k, info in result.get("responders", {}).items():
        timeline = info.get("timeline")
        if timeline:
            segs[k] = [
                (entry["type"], entry["from"], entry["to"], entry["t0"], entry["t1"])
                for entry in timeline
            ]
            continue
        path = info.get("path", [])
        t = 0.0
        ksegs = []
        for (u, v) in path:
            # walking segment
            if v.startswith("G_") or u.startswith("G_"):
                # room to nearest exit
                node = u if v.startswith("G_") else v
                # find nearest exit
                exits = set(start_node.values())
                dmin = min(
                    ((coords[node][0] - coords[e][0]) ** 2 + (coords[node][1] - coords[e][1]) ** 2) ** 0.5
                    for e in exits
                )
                dt_walk = dmin / max(walk_speed, 1e-6)
                ksegs.append(("walk", u, v, t, t + dt_walk))
                t += dt_walk
            else:
                dx = coords[u][0] - coords[v][0]
                dy = coords[u][1] - coords[v][1]
                dt_walk = ((dx * dx + dy * dy) ** 0.5) / max(walk_speed, 1e-6)
                ksegs.append(("walk", u, v, t, t + dt_walk))
                t += dt_walk

            # clearing segment if arriving in a room
            if v in rooms:
                dt_clear = sweep_time.get(v, 0.0)
                ksegs.append(("clear", v, v, t, t + dt_clear))
                t += dt_clear
        segs[k] = ksegs

    # Plot Gantt
    fig, ax = plt.subplots(figsize=(10, 3 + 0.8 * len(segs)))
    base_palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    def _variant(color, typ):
        rgb = mcolors.to_rgb(color)
        if typ == "clear":
            rgb = tuple(min(1.0, c + 0.4 * (1 - c)) for c in rgb)
        elif typ == "escort":
            rgb = tuple(min(1.0, c + 0.15) for c in rgb)
        elif typ == "return":
            rgb = tuple(max(0.0, c * 0.6) for c in rgb)
        else:
            rgb = tuple(max(0.0, c * 0.85) for c in rgb)
        return rgb
    yticks = []
    yticklbls = []
    y = 10
    legends = []

    for idx, (k, ksegs) in enumerate(segs.items()):
        base = base_palette[idx % len(base_palette)]
        for typ, u, v, t0, t1 in ksegs:
            color = _variant(base, typ)
            ax.broken_barh([(t0, t1 - t0)], (y, 8), facecolors=color, edgecolors="none", alpha=0.95)
            if typ == "clear":
                ax.text(t0 + (t1 - t0) / 2, y + 4, v, ha="center", va="center", fontsize=8, color="#111")
        yticks.append(y + 4)
        yticklbls.append(k)
        legends.append((base, k))
        y += 15

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklbls)
    ax.set_xlabel("Time (s) [labels show room for clear segments]")
    ax.set_title("Responder Timeline (walk vs clear)", pad=16)
    # Nice HH:MM:SS tick labels
    def secfmt(x, pos=None):
        return _fmt_hms(x)
    import matplotlib.ticker as mticker
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(secfmt))
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    ax.set_xlim(
        0,
        max(
            result.get("T_total", 0.0),
            max((t1 for ksegs in segs.values() for (_, _, _, _, t1) in ksegs), default=0.0),
        ),
    )
    # custom legend
    handles = []
    for base, label in legends:
        for typ in ("walk", "escort", "return", "clear"):
            handles.append(
                plt.Line2D([0], [0], color=_variant(base, typ), lw=6, label=f"{label} {typ}"),
            )
    ax.legend(handles=handles, loc="upper right", fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)


def animate_gif(
    coords: Dict[str, Tuple[float, float]],
    rooms: List[str],
    start_node: Dict[str, str],
    result: dict,
    savepath: str = "out/anim.gif",
    fps: int = 24,
    duration_sec: int = 12,
    figsize=(10, 5),
    dpi: int = 140,
):
    """Animate responder movement and clearing as a GIF (cleaner styling).

    Improvements vs. previous:
      - Fixed total duration (duration_sec) for consistent playback
      - Smoother 24 fps, higher DPI, cleaner theme
      - Faint full routes + growing trail of movement per responder
      - Larger markers with white outline; rooms glow green upon first clear
      - Subtle grid, hidden spines, readable time overlay
    """
    _ensure_dir(savepath)
    import numpy as np
    from matplotlib.animation import FuncAnimation, PillowWriter

    walk_speed = float(result.get("meta", {}).get("walk_speed_used", 1.0))
    sweep_time = result.get("sweep_time", {})
    exits = set(result.get("meta", {}).get("exits", list(start_node.values())))

    # Build segments per responder: (typ, u, v, p0, p1, t0, t1)
    segs = {}
    for k, info in result.get("responders", {}).items():
        timeline = info.get("timeline")
        if timeline:
            ksegs = [
                (
                    entry["type"],
                    entry["from"],
                    entry["to"],
                    entry["p0"],
                    entry["p1"],
                    entry["t0"],
                    entry["t1"],
                )
                for entry in timeline
            ]
            segs[k] = ksegs
            continue
        path = info.get("path", [])
        t = 0.0
        ksegs = []
        for (u, v) in path:
            # walking segment
            if v.startswith("G_") or u.startswith("G_"):
                node = u if v.startswith("G_") else v
                # nearest exit target
                e_best = None
                d_best = 1e18
                for e in exits:
                    dx = coords[node][0] - coords[e][0]
                    dy = coords[node][1] - coords[e][1]
                    d = (dx * dx + dy * dy) ** 0.5
                    if d < d_best:
                        d_best = d
                        e_best = e
                p0 = coords[node]
                p1 = coords[e_best]
                dt_walk = d_best / max(walk_speed, 1e-6)
            else:
                p0 = coords[u]
                p1 = coords[v]
                dx = p0[0] - p1[0]
                dy = p0[1] - p1[1]
                dt_walk = ((dx * dx + dy * dy) ** 0.5) / max(walk_speed, 1e-6)
            ksegs.append(("walk", u, v, p0, p1, t, t + dt_walk))
            t += dt_walk

            # clearing segment if arriving at a room
            if v in rooms:
                dt_clear = float(sweep_time.get(v, 0.0))
                ksegs.append(("clear", v, v, p1, p1, t, t + dt_clear))
                t += dt_clear
        segs[k] = ksegs

    # Determine total duration
    total_t = max((t1 for ksegs in segs.values() for (_, _, _, _, _, _, t1) in ksegs), default=0.0)
    if total_t <= 0:
        # Nothing to animate
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No animation (empty path)", ha="center", va="center")
        fig.savefig(savepath, dpi=120)
        plt.close(fig)
        return

    # Build room first-clear times
    fc = result.get("first_clear", {}).get("by_room", {})
    room_first = {r: (fc.get(r, {}) or {}).get("time") for r in rooms}

    # Frame scheduling: compress to fixed duration_sec @ fps
    n_frames = max(1, int(fps * duration_sec))
    dt = total_t / n_frames

    # Prepare figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # Static rooms
    room_xy = np.array([[coords[r][0], coords[r][1]] for r in rooms])
    room_colors = np.array([[0.6, 0.6, 0.6, 0.65] for _ in rooms])  # soft gray before clear
    rooms_sc = ax.scatter(room_xy[:, 0], room_xy[:, 1], marker="s", s=90, c=room_colors, edgecolors="#dddddd", linewidths=0.8)
    # Exits
    ex = np.array([[coords[e][0], coords[e][1]] for e in exits])
    ax.scatter(ex[:, 0], ex[:, 1], marker="^", s=90, c="#2ca02c")
    # Labels
    for r in rooms:
        ax.text(coords[r][0] + 0.8, coords[r][1] + 0.3, r, fontsize=9, color="#333")
    for e in exits:
        ax.text(
            coords[e][0] + 0.8,
            coords[e][1] + 0.3,
            e,
            fontsize=8,
            color="#2ca02c",
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.6, edgecolor="none"),
        )

    # Responder markers
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    responders = list(result.get("responders", {}).keys())
    dots = {}
    trails = {}
    full_routes = {}
    for idx, k in enumerate(responders):
        c = colors[idx % len(colors)]
        # Build full route polyline for context
        seq = []
        s = start_node[k]
        seq.append(coords[s])
        for (u, v) in result.get("responders", {}).get(k, {}).get("path", []):
            if v.startswith("G_"):
                # map to nearest exit from u
                e_best = None
                d_best = 1e18
                for e in exits:
                    dx = coords[u][0] - coords[e][0]
                    dy = coords[u][1] - coords[e][1]
                    d = (dx * dx + dy * dy) ** 0.5
                    if d < d_best:
                        d_best = d
                        e_best = e
                seq.append(coords[e_best])
            else:
                seq.append(coords[v])
        xs = [p[0] for p in seq]
        ys = [p[1] for p in seq]
        full_routes[k], = ax.plot(xs, ys, color=c, alpha=0.25, lw=2)
        # dynamic trail
        trails[k], = ax.plot([xs[0]], [ys[0]], color=c, alpha=0.85, lw=3)
        # moving marker with outline
        dots[k] = ax.scatter([xs[0]], [ys[0]], s=90, c=c, edgecolors="#ffffff", linewidths=1.5, label=k, zorder=5)

    # style axes
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.3)

    # time label
    time_text = ax.text(0.01, 0.98, "t=00:00:00", transform=ax.transAxes, ha="left", va="top", fontsize=11,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.7))

    # set limits
    xs = [v[0] for v in coords.values()]
    ys = [v[1] for v in coords.values()]
    pad_x = (max(xs) - min(xs)) * 0.05 + 2
    pad_y = (max(ys) - min(ys)) * 0.1 + 2
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("GraphEvac Animation")
    ax.legend(loc="upper center", ncol=5, fontsize=8, frameon=False)

    def interp_pos(t_now: float, k: str):
        ksegs = segs.get(k, [])
        if not ksegs:
            return None
        # Before first segment
        if t_now <= ksegs[0][5]:
            return ksegs[0][3]
        for (typ, u, v, p0, p1, t0, t1) in ksegs:
            if t_now < t0:
                return p0
            if t0 <= t_now <= t1:
                if typ == "walk" and t1 > t0:
                    a = (t_now - t0) / (t1 - t0)
                    return (p0[0] * (1 - a) + p1[0] * a, p0[1] * (1 - a) + p1[1] * a)
                else:
                    return p1
        # After last segment
        return ksegs[-1][4]

    # pre-identify rooms with ongoing clear at time t
    # Build quick lookup: for each responder, list of (room, t0, t1)
    clear_spans = {}
    for k, ksegs in segs.items():
        spans = []
        for (typ, u, v, p0, p1, t0, t1) in ksegs:
            if typ == "clear" and v in rooms:
                spans.append((v, t0, t1))
        clear_spans[k] = spans

    # concentric glow handles for current clearing rooms
    glow_art = {}

    def update(frame_idx):
        t_now = frame_idx * dt
        time_text.set_text(f"t={_fmt_hms(t_now)}")
        # update room colors by first-clear
        for i, r in enumerate(rooms):
            t_clear = room_first.get(r)
            if t_clear is not None and t_now >= t_clear:
                room_colors[i] = [0.1, 0.7, 0.2, 0.95]  # vivid green
        rooms_sc.set_facecolors(room_colors)

        # remove previous glow
        for art in glow_art.values():
            art.remove()
        glow_art.clear()

        # responders: update position and trail; add glow for clearing
        for k in responders:
            pos = interp_pos(t_now, k)
            if pos is None:
                continue
            # update marker
            dots[k].set_offsets([[pos[0], pos[1]]])
            # extend trail
            xd = trails[k].get_xdata()
            yd = trails[k].get_ydata()
            trails[k].set_data(list(xd) + [pos[0]], list(yd) + [pos[1]])

            # glow rings for current clearing
            for (r, t0, t1) in clear_spans.get(k, []):
                if t0 <= t_now <= t1:
                    prog = 0 if t1 == t0 else (t_now - t0) / (t1 - t0)
                    rad = 2.0 + 4.0 * prog
                    glow_art[(k, r)] = ax.scatter([coords[r][0]], [coords[r][1]], s=300 * rad, facecolors='none',
                                                  edgecolors='#2ca02c', linewidths=1.2, alpha=0.5, zorder=4)

        time_text.set_text(f"t={_fmt_hms(t_now)}")
        return [rooms_sc, time_text] + [dots[k] for k in responders] + list(trails.values()) + list(glow_art.values())

    anim = FuncAnimation(fig, update, frames=n_frames, interval=int(1000 / max(fps, 1)), blit=False)
    writer = PillowWriter(fps=fps)
    anim.save(savepath, writer=writer)
    plt.close(fig)
