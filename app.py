"""
Chopsticks SPE Explorer — Streamlit App
========================================
Run with:  streamlit run app.py
"""

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np
from collections import defaultdict
import random

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Chopsticks SPE Explorer",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3, h4 { font-family: 'IBM Plex Sans', sans-serif; font-weight: 600; }

.hero-box {
    background: linear-gradient(135deg, #0f2027 0%, #1a3a4a 50%, #203a43 100%);
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    border: 1px solid #2a5a6a;
}
.hero-box h1 { color: #e0f4f1; font-size: 2.2rem; margin: 0 0 .5rem; }
.hero-box p  { color: #a8d8d0; font-size: 1.05rem; margin: 0; line-height: 1.6; }

.verdict-win  { background:#e8f8f2; border-left:5px solid #1D9E75; padding:1rem 1.2rem; border-radius:8px; margin:.8rem 0; }
.verdict-lose { background:#fff0eb; border-left:5px solid #D85A30; padding:1rem 1.2rem; border-radius:8px; margin:.8rem 0; }
.verdict-tie  { background:#f0effc; border-left:5px solid #7F77DD; padding:1rem 1.2rem; border-radius:8px; margin:.8rem 0; }

.badge-win  { background:#1D9E75; color:#fff; padding:3px 10px; border-radius:12px; font-size:.8rem; font-weight:600; }
.badge-lose { background:#D85A30; color:#fff; padding:3px 10px; border-radius:12px; font-size:.8rem; font-weight:600; }
.badge-tie  { background:#7F77DD; color:#fff; padding:3px 10px; border-radius:12px; font-size:.8rem; font-weight:600; }

.hand-display {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem; letter-spacing: 4px;
    text-align: center; padding: .6rem 0;
}
.hand-dead { color: #aaa; text-decoration: line-through; }
.hand-alive { color: #1a1a1a; }

.state-box {
    border: 2px solid #dee; border-radius:10px; padding:1rem;
    background:#fafafa; text-align:center;
}
.info-callout {
    background:#f0f4f8; border-radius:10px; padding:1rem 1.3rem;
    border:1px solid #cdd8e3; margin:.8rem 0; font-size:.93rem;
    line-height:1.65;
}
.mono { font-family:'IBM Plex Mono', monospace; font-size:.88rem; }
.step-strip { display:flex; overflow-x:auto; gap:10px; padding:.5rem 0; }
.step-card {
    min-width:110px; border-radius:10px; padding:.7rem .5rem;
    text-align:center; border:1px solid #ddd; font-size:.8rem;
    flex-shrink:0;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 1. GAME ENGINE
# ══════════════════════════════════════════════════════════════

class BadMove(Exception):
    pass

def normalize(pos):
    a, b, c, d = pos
    a %= 5; b %= 5; c %= 5; d %= 5
    if b > a: a, b = b, a
    if d > c: c, d = d, c
    return (a, b, c, d)

def next_turn(pos):
    a, b, c, d = pos
    return (c, d, a, b)

def move_tap(pos, use_left, tap_left):
    a, b, c, d = pos
    src = a if use_left else b
    if src == 0:           raise BadMove("Hand is dead")
    if tap_left  and c==0: raise BadMove("Target hand is dead")
    if not tap_left and d==0: raise BadMove("Target hand is dead")
    if tap_left:  c += src
    else:         d += src
    return next_turn(normalize((a, b, c, d)))

@st.cache_data
def get_all_states_and_nexts():
    """Build full reachability graph from (1,1,1,1). Cached."""
    nexts_map = {}

    def compute_nexts(pos):
        if pos in nexts_map: return nexts_map[pos]
        a, b, c, d = pos
        if (a==0 and b==0) or (c==0 and d==0):
            nexts_map[pos] = []
            return []
        result = set()
        for ul in (True, False):
            for tl in (True, False):
                try: result.add(move_tap(pos, ul, tl))
                except BadMove: pass
        nexts_map[pos] = sorted(result)
        return nexts_map[pos]

    start = (1,1,1,1)
    visited = {start}
    frontier = [start]
    while frontier:
        nxt = []
        for p in frontier:
            for n in compute_nexts(p):
                if n not in visited:
                    visited.add(n); nxt.append(n)
        frontier = nxt

    # Ensure all states have entries
    for s in visited:
        compute_nexts(s)

    return sorted(visited), nexts_map

@st.cache_data
def get_classification():
    """Run backward induction. Returns (winning, losing, tying, recommended, win_depth)."""
    all_states, nexts_map = get_all_states_and_nexts()
    all_set = set(all_states)

    term_losing = set(s for s in all_states if s[0]==0 and s[1]==0 and (s[2]>0 or s[3]>0))
    losing  = set(term_losing)
    winning = set()

    changed = True
    while changed:
        changed = False
        for s in all_states:
            if s in winning or s in losing: continue
            if any(n in losing for n in nexts_map[s]):
                winning.add(s); changed = True
        for s in all_states:
            if s in winning or s in losing: continue
            ns = nexts_map[s]
            if ns and all(n in winning for n in ns):
                losing.add(s); changed = True

    tying = all_set - winning - losing

    # BFS depth
    win_depth = {}
    queue = list(term_losing)
    for s in term_losing: win_depth[s] = 0
    while queue:
        s = queue.pop(0)
        for p in all_states:
            if s in nexts_map.get(p, []) and p not in win_depth:
                win_depth[p] = win_depth[s]+1
                queue.append(p)

    # Recommended moves
    recommended = {}
    for s in all_states:
        ns = nexts_map[s]
        if s in winning:
            lm = sorted([n for n in ns if n in losing], key=lambda x: win_depth.get(x,999))
            recommended[s] = lm
        elif s in losing:
            recommended[s] = []
        else:
            tm = [n for n in ns if n in tying]
            recommended[s] = tm if tm else ns

    return winning, losing, tying, recommended, win_depth

def classify_state(s, winning, losing):
    if s in winning: return "win"
    if s in losing:  return "lose"
    return "tie"

def fmt(pos):
    return f"({','.join(map(str,pos))})"

WIN_C  = "#1D9E75"
LOSE_C = "#D85A30"
TIE_C  = "#7F77DD"
BG_C   = "#FAFAF8"
P_BG   = "#F4F3EE"

def state_color(s, winning, losing):
    if s in winning: return WIN_C
    if s in losing:  return LOSE_C
    return TIE_C

def badge(s, winning, losing):
    t = classify_state(s, winning, losing)
    labels = {"win":"WIN ✓","lose":"LOSE ✗","tie":"DRAW ↺"}
    return f'<span class="badge-{t}">{labels[t]}</span>'

def hand_html(n):
    if n == 0:
        return '<span class="hand-dead">✕</span>'
    fingers = "●" * n + "○" * (4-n)
    return f'<span class="hand-alive">{fingers}</span>'


# ══════════════════════════════════════════════════════════════
# 2. SIMULATION
# ══════════════════════════════════════════════════════════════

def simulate_game(start, recommended, nexts_map, max_steps=60):
    path = [(start, "Start")]
    pos  = start
    for step in range(max_steps):
        moves = recommended.get(pos, nexts_map.get(pos, []))
        if not moves: break
        pos = moves[0]
        path.append((pos, f"{'P1' if step%2==0 else 'P2'} moved"))
        if pos[0]==0 and pos[1]==0: return path, "P2 wins 🎉"
        if pos[2]==0 and pos[3]==0: return path, "P1 wins 🎉"
    return path, "Draw — infinite cycle ↺"


# ══════════════════════════════════════════════════════════════
# 3. MATPLOTLIB FIGURES
# ══════════════════════════════════════════════════════════════

def fig_overview(winning, losing, tying, win_depth, all_states):
    fig = plt.figure(figsize=(15, 8), facecolor=BG_C)
    gs  = gridspec.GridSpec(2,3, figure=fig, hspace=.46, wspace=.38,
                            left=.06, right=.97, top=.88, bottom=.09)

    # Pie
    ax0 = fig.add_subplot(gs[0,0])
    sz  = [len(winning), len(losing), len(tying)]
    lbl = [f"Winning\n{sz[0]}", f"Losing\n{sz[1]}", f"Tying\n{sz[2]}"]
    _, _, autos = ax0.pie(sz, labels=lbl, colors=[WIN_C,LOSE_C,TIE_C],
                          autopct="%1.0f%%", startangle=90,
                          textprops={"fontsize":9},
                          wedgeprops={"linewidth":1.5,"edgecolor":BG_C})
    for a in autos: a.set(fontsize=8, color="white", fontweight="bold")
    ax0.set_title("All States\nclassified", fontsize=10, pad=8)

    # Win-depth hist
    ax1 = fig.add_subplot(gs[0,1])
    wdv = [win_depth[s] for s in winning if s in win_depth]
    ax1.hist(wdv, bins=range(1,max(wdv)+2), color=WIN_C, edgecolor=BG_C, alpha=.85)
    ax1.set(xlabel="Depth (moves to force win)", ylabel="# states",
            title="Winning State Depths")
    ax1.set_facecolor(P_BG); ax1.tick_params(labelsize=8)

    # Lose-depth hist
    ax2 = fig.add_subplot(gs[0,2])
    ldv = [win_depth[s] for s in losing if s in win_depth]
    if ldv:
        ax2.hist(ldv, bins=range(0,max(ldv)+2), color=LOSE_C, edgecolor=BG_C, alpha=.85)
    ax2.set(xlabel="Depth from terminal loss", ylabel="# states",
            title="Losing State Depths")
    ax2.set_facecolor(P_BG); ax2.tick_params(labelsize=8)

    # Scatter
    ax3 = fig.add_subplot(gs[1,:])
    for s in tying:   ax3.scatter(s[0]+s[1],s[2]+s[3],color=TIE_C, alpha=.4, s=25, zorder=2)
    for s in winning: ax3.scatter(s[0]+s[1],s[2]+s[3],color=WIN_C, alpha=.65,s=36, zorder=3)
    for s in losing:  ax3.scatter(s[0]+s[1],s[2]+s[3],color=LOSE_C,alpha=.85,s=48, zorder=4,marker="X")
    ax3.set(xlabel="P1 total fingers", ylabel="P2 total fingers",
            title="State Map: P1 total vs P2 total  (✕ = losing, • = winning, · = tying)")
    ax3.set_facecolor(P_BG); ax3.tick_params(labelsize=9)
    ax3.set_xticks(range(9)); ax3.set_yticks(range(9))
    ax3.legend(handles=[
        mpatches.Patch(color=WIN_C,  label=f"Winning ({sz[0]})"),
        mpatches.Patch(color=LOSE_C, label=f"Losing ({sz[1]})"),
        mpatches.Patch(color=TIE_C,  label=f"Tying ({sz[2]})"),
    ], fontsize=8, loc="upper right")
    fig.suptitle("Chopsticks — Complete SPE Classification of All 200 Reachable States",
                 fontsize=13, fontweight="bold", color="#2C2C2A")
    return fig


def fig_heatmap(all_states, winning, losing, tying):
    cell = defaultdict(lambda:{"W":0,"L":0,"T":0})
    for s in all_states:
        k=(s[0]+s[1], s[2]+s[3])
        if s in winning: cell[k]["W"]+=1
        elif s in losing: cell[k]["L"]+=1
        else: cell[k]["T"]+=1

    fig, axes = plt.subplots(1,3, figsize=(15,5), facecolor=BG_C)
    fig.suptitle("Heatmaps: count of W/L/T states per (P1 total, P2 total) cell",
                 fontsize=12, fontweight="bold")
    for ax,(title,key,cmap) in zip(axes,[("Winning","W","Greens"),
                                          ("Losing","L","Reds"),
                                          ("Tying","T","Purples")]):
        g = np.array([[cell.get((r,c),{}).get(key,0) for c in range(9)] for r in range(9)])
        im = ax.imshow(g, cmap=cmap, origin="lower", aspect="auto", vmin=0, vmax=max(1,g.max()))
        ax.set(title=title, xlabel="P2 total", ylabel="P1 total")
        ax.set_xticks(range(9)); ax.set_yticks(range(9)); ax.tick_params(labelsize=8)
        for i in range(9):
            for j in range(9):
                v=g[i,j]
                if v>0:
                    ax.text(j,i,str(int(v)),ha="center",va="center",fontsize=8,
                            color="white" if v>2 else "#333", fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=.75)
    plt.tight_layout()
    return fig


def fig_game_tree(start, winning, losing, tying, recommended, nexts_map, depth=3):
    G = nx.DiGraph()
    nc_map, lbl_map, rec_edges = {}, {}, set()

    def color(s):
        if s in winning: return WIN_C
        if s in losing:  return LOSE_C
        return TIE_C

    def bfs(pos, d):
        if d==0: return
        G.add_node(pos); nc_map[pos]=color(pos); lbl_map[pos]=fmt(pos)
        for n in nexts_map.get(pos,[]):
            G.add_node(n); nc_map[n]=color(n); lbl_map[n]=fmt(n)
            G.add_edge(pos,n)
            if n in recommended.get(pos,[]): rec_edges.add((pos,n))
            bfs(n,d-1)

    bfs(start, depth)
    fig, ax = plt.subplots(figsize=(16,9), facecolor=BG_C)
    ax.set_facecolor(BG_C)
    ax.set_title(f"SPE Game Tree from {fmt(start)} — {depth} levels deep\n"
                 "Thick green = SPE-recommended move", fontsize=12, pad=10)

    try:    pos_l = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except: pos_l = nx.spring_layout(G, seed=42, k=2.2)

    nl = list(G.nodes())
    nx.draw_networkx_nodes(G,pos_l,nodelist=nl,
                           node_color=[nc_map.get(n,TIE_C) for n in nl],
                           node_size=800, ax=ax, alpha=.9)
    nx.draw_networkx_labels(G,pos_l,labels={n:lbl_map.get(n,"") for n in nl},
                            font_size=6.5,font_color="white",font_weight="bold",ax=ax)
    regular = [(u,v) for u,v in G.edges() if (u,v) not in rec_edges]
    nx.draw_networkx_edges(G,pos_l,edgelist=regular,edge_color="#aaa",alpha=.45,
                           arrows=True,arrowsize=12,connectionstyle="arc3,rad=0.1",ax=ax)
    nx.draw_networkx_edges(G,pos_l,edgelist=list(rec_edges),edge_color=WIN_C,
                           width=2.6,alpha=.9,arrows=True,arrowsize=16,
                           connectionstyle="arc3,rad=0.1",ax=ax)
    ax.legend(handles=[
        mpatches.Patch(color=WIN_C, label="Winning"),
        mpatches.Patch(color=LOSE_C,label="Losing"),
        mpatches.Patch(color=TIE_C, label="Tying"),
    ], fontsize=9, loc="upper right")
    ax.axis("off")
    plt.tight_layout()
    return fig


def fig_path_strip(path, winning, losing, n_show=12):
    path = path[:n_show]
    n = len(path)
    fig, axes = plt.subplots(1, n, figsize=(max(12, n*1.5), 3.6), facecolor=BG_C)
    if n==1: axes=[axes]

    def sc(s):
        if s in winning: return WIN_C
        if s in losing:  return LOSE_C
        return TIE_C
    def sl(s):
        if s in winning: return "WIN"
        if s in losing:  return "LOSE"
        return "TIE"

    for i,(state,who) in enumerate(path):
        ax=axes[i]; a,b,c,d=state; col=sc(state)
        ax.set_facecolor(col+"18"); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
        ax.text(.5,.96,f"Step {i}",ha="center",va="top",fontsize=7.5,
                color="#777",transform=ax.transAxes)
        ax.text(.5,.86,who,ha="center",va="top",fontsize=7,
                color="#555",transform=ax.transAxes,style="italic")
        ax.text(.5,.74,"P1",ha="center",fontsize=9,fontweight="bold",
                color="#222",transform=ax.transAxes)
        p1s="×|×" if a+b==0 else f"{'●'*a}{'○'*(4-a)}|{'●'*b}{'○'*(4-b)}"
        ax.text(.5,.62,p1s,ha="center",fontsize=8,
                color=LOSE_C if a+b==0 else "#222",
                transform=ax.transAxes,family="monospace")
        ax.axhline(.54,color="#ddd",lw=.8)
        ax.text(.5,.47,"P2",ha="center",fontsize=9,fontweight="bold",
                color="#222",transform=ax.transAxes)
        p2s="×|×" if c+d==0 else f"{'●'*c}{'○'*(4-c)}|{'●'*d}{'○'*(4-d)}"
        ax.text(.5,.35,p2s,ha="center",fontsize=8,
                color=LOSE_C if c+d==0 else "#222",
                transform=ax.transAxes,family="monospace")
        ax.text(.5,.17,sl(state),ha="center",va="center",fontsize=8,
                fontweight="bold",color="white",transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=.3",facecolor=col,edgecolor="none"))
        ax.text(.5,.04,fmt(state),ha="center",va="bottom",fontsize=6.5,
                color="#999",transform=ax.transAxes)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# 4. SIDEBAR
# ══════════════════════════════════════════════════════════════

def sidebar():
    with st.sidebar:
        st.markdown("## ✋ Chopsticks SPE")
        st.markdown("""
**Ruleset (no rollover)**
- Each player has two hands starting at **1 finger**
- On your turn: tap one of your hands onto one of the opponent's live hands — their count increases by yours
- **≥ 5 fingers → hand dies** (no modulo, just eliminated)
- A player **loses when both hands are dead**
- No splits or transfers (ALLOW_SWAP = False)
""")
        st.divider()
        page = st.radio("Navigate",
                        ["🏠 Introduction",
                         "🔬 SPE Analysis",
                         "📊 Visualizations",
                         "🌲 Game Tree",
                         "🎮 Play the Game",
                         "📋 Full State Table"],
                        label_visibility="collapsed")
        st.divider()
        st.caption("Built with Python · Matplotlib · NetworkX · Streamlit")
    return page


# ══════════════════════════════════════════════════════════════
# 5. PAGES
# ══════════════════════════════════════════════════════════════

def page_intro():
    st.markdown("""
<div class="hero-box">
  <h1>✋ Chopsticks — SPE Explorer</h1>
  <p>An interactive deep-dive into the <strong>Subgame Perfect Equilibrium</strong> of Chopsticks
  under the no-rollover rule. Who wins with perfect play? Navigate the pages on the left to explore.</p>
  <p style="margin-top:1rem;font-size:.92rem;color:#7ecfc5;border-top:1px solid #2a5a6a;padding-top:.8rem">
    Built by <strong style="color:#e0f4f1">Erfan Zarafshan</strong> &nbsp;&middot;&nbsp;
    PhD Student in Economics &nbsp;&middot;&nbsp;
    <em>Louisiana State University</em>
  </p>
</div>
""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("What is Chopsticks?")
        st.markdown("""
Chopsticks is a two-player hand game. Each player starts with **1 finger** raised on each hand.

**On your turn you must:**
1. Pick one of your *live* hands (> 0 fingers)
2. Tap it against one of the *opponent's* live hands
3. Their tapped hand gains your fingers

**A hand dies** the moment it reaches **5 or more** fingers (in this version, no modulo wraparound — it's just eliminated).

**You lose** when both your hands are dead (0,0).
        """)
        st.markdown("""
**Example:**
- You have hands `(3, 1)`, opponent has `(2, 1)`
- You tap your `3` onto their `2` → their hand becomes `5` → **killed** → `(0, 1)`
- Now they have only one live hand left
        """)

    with c2:
        st.subheader("What is SPE?")
        st.markdown("""
A **Subgame Perfect Equilibrium (SPE)** is a solution concept in game theory where players
choose optimally at *every* decision node — not just the overall start, but in every possible
situation they might find themselves in.

**How it's computed here:**

1. **Base case:** Any position where your hands are both 0 is a *loss* for you
2. **Backward induction:**
   - A position is **Winning** if you can move to a Losing position for the opponent
   - A position is **Losing** if *every* move you make gives the opponent a Winning position
   - A position is **Tying** if none of the above apply — the game cycles
3. The **SPE strategy** is: from a Winning state, always move to the deepest Losing state available.
   From a Tying state, always stay within Tying states (never hand the opponent a win).

**Key result:** The starting position `(1,1,1,1)` is a **Tying state** — with perfect play from both sides, the game **never ends**.
        """)

    st.divider()
    st.subheader("At a Glance")
    all_states, nexts_map = get_all_states_and_nexts()
    winning, losing, tying, recommended, win_depth = get_classification()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total reachable states", len(all_states))
    m2.metric("Winning states", len(winning), help="Current player can force a win")
    m3.metric("Losing states",  len(losing),  help="Current player will lose with perfect opponent")
    m4.metric("Tying states",   len(tying),   help="Neither player can force a win — draw")

    st.markdown("""
<div class="verdict-tie">
<strong>✋ Main Result:</strong> Starting from <code>(1,1,1,1)</code>, the game is a <strong>Draw</strong>.
Neither player has a guaranteed winning strategy. With SPE play, the game cycles through the 72 tying states indefinitely.
The only SPE move for P1 from the start is to tap → <code>(2,1,1,1)</code>.
</div>
""", unsafe_allow_html=True)


def page_analysis():
    st.header("🔬 SPE Analysis — Full Breakdown")
    all_states, nexts_map = get_all_states_and_nexts()
    winning, losing, tying, recommended, win_depth = get_classification()

    st.subheader("1 · How Backward Induction Works")
    st.markdown("""
The algorithm classifies every reachable state into one of three categories using **iterated elimination**:
    """)

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("""
<div class="verdict-lose">
<strong>🔴 Losing state</strong><br>
Every move you can make leads to a Winning position for your opponent.
You are trapped — there is no escape.
<br><br>
<em>Base case:</em> any state <code>(0,0,c,d)</code> where you have no live hands.
</div>
""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div class="verdict-win">
<strong>🟢 Winning state</strong><br>
You have at least one move that lands the opponent in a Losing state.
Play that move and win.
<br><br>
<em>Depth:</em> how many moves until the opponent's last hand dies ranges from 1 to 5.
</div>
""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
<div class="verdict-tie">
<strong>🟣 Tying state</strong><br>
Neither player can be forced into a loss. The game can cycle indefinitely.
Both players' SPE strategy: stay within tying states forever.
<br><br>
<em>Starting position <code>(1,1,1,1)</code> is here.</em>
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.subheader("2 · The Starting Position")

    start = (1,1,1,1)
    cls = classify_state(start, winning, losing)
    recs = recommended.get(start, [])

    col1, col2 = st.columns([1,2])
    with col1:
        st.markdown(f"""
<div class="state-box">
<div style="font-size:1.1rem;font-weight:600;margin-bottom:.5rem">Starting state</div>
<div class="mono">{fmt(start)}</div>
<br>
{badge(start, winning, losing)}
<br><br>
<div style="font-size:.9rem;color:#555">
SPE move(s):<br>
{"<br>".join(f'<code>{fmt(r)}</code>' for r in recs) if recs else "none"}
</div>
</div>
""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
The starting position `(1,1,1,1)` — both players have both hands alive with 1 finger each —
is a **tying state**. This means:

- Player 1 cannot force a win
- Player 2 cannot force a win
- With perfect (SPE) play from both, the game cycles infinitely

Player 1's **only** SPE-consistent move from `(1,1,1,1)` is to tap their 1-finger hand onto
one of Player 2's 1-finger hands, giving `(2,1,1,1)` after normalization.
Any deviation risks entering a losing position.
        """)

    st.divider()
    st.subheader("3 · Winning State Distances")
    st.markdown("""
Among the 82 winning states, each has a **depth** — the minimum number of steps needed for the winning player
to force the opponent's both hands to zero. Depth 1 means you can kill both opponent hands in one tap.
    """)

    by_depth = defaultdict(list)
    for s in winning:
        by_depth[win_depth.get(s,-1)].append(s)

    for d in sorted(by_depth.keys()):
        states = sorted(by_depth[d])
        with st.expander(f"Depth {d}  —  {len(states)} winning states"):
            rows = []
            for s in states:
                r = recommended.get(s,[])
                rows.append({
                    "State": fmt(s),
                    "P1 hands": f"{s[0]},{s[1]}",
                    "P2 hands": f"{s[2]},{s[3]}",
                    "SPE move →": fmt(r[0]) if r else "—"
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("4 · SPE Strategy Profile (summary)")
    st.markdown("""
The complete SPE strategy profile is a mapping from every state to the recommended action:

| State type | SPE action | Reasoning |
|---|---|---|
| **Winning** (82 states) | Move to the deepest reachable Losing state | Forces opponent onto a losing path; deeper = more forced |
| **Losing** (46 states) | Any move (all lead to Winning for opponent) | No escape; you will lose regardless |
| **Tying** (72 states) | Stay within Tying states; avoid Losing states | Prevents opponent from ever reaching a Winning position |

The equilibrium is **unique in outcome** (always a draw from start) but **not unique in strategy** —
there are often multiple SPE moves from tying states (any move that stays within the tying set works).
    """)


def page_visualizations():
    st.header("📊 Visualizations")
    all_states, nexts_map = get_all_states_and_nexts()
    winning, losing, tying, recommended, win_depth = get_classification()

    tab1, tab2, tab3 = st.tabs(["Overview", "Heatmaps", "Optimal Play Path"])

    with tab1:
        st.subheader("Full State Classification Overview")
        st.markdown("""
Four panels showing how all 200 reachable states distribute across the three categories.

- **Pie chart** (top left): raw counts — 82 winning, 46 losing, 72 tying
- **Depth histograms** (top center/right): how "far" winning/losing states are from terminal positions
- **Scatter** (bottom): every state plotted by P1's total fingers vs P2's total fingers,
  colored by classification. Notice losing states (✕) cluster when one player has very
  few fingers total.
        """)
        fig = fig_overview(winning, losing, tying, win_depth, all_states)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab2:
        st.subheader("Classification Heatmaps")
        st.markdown("""
Each cell `(row=P1 total, col=P2 total)` shows how many states of that type exist.
Multiple states can share the same totals but differ in how fingers are *distributed*
across the two hands (e.g., `(3,1)` vs `(2,2)` both total 4).

Key insight: **Losing states** dominate when P2 has many more total fingers than P1 (top-left area of the Losing map).
**Winning states** cluster in the opposite corner.
        """)
        fig = fig_heatmap(all_states, winning, losing, tying)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab3:
        st.subheader("SPE Optimal Play Path from (1,1,1,1)")
        st.markdown("""
Each card below shows one step of SPE play. Both players always pick the SPE-recommended move.
Because `(1,1,1,1)` is a tying state, the game cycles — the strip shows the first 12 steps.

- **●** = live finger, **○** = empty slot
- Badge color: 🟢 Winning · 🔴 Losing · 🟣 Tying
        """)
        path, outcome = simulate_game((1,1,1,1), recommended, nexts_map, max_steps=12)
        st.markdown(f"**Outcome (after 12 steps of SPE play):** `{outcome}`")
        fig = fig_path_strip(path, winning, losing, n_show=12)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("""
<div class="info-callout">
<strong>Why does it cycle?</strong> The 72 tying states form a strongly connected component —
every tying state can reach another tying state, and SPE instructs both players to <em>never</em>
leave this set. There is no "exit" that doesn't either put you in a losing position or give the
opponent a winning position. So both players rationally orbit these states forever.
</div>
""", unsafe_allow_html=True)


def page_game_tree():
    st.header("🌲 SPE Game Tree")
    all_states, nexts_map = get_all_states_and_nexts()
    winning, losing, tying, recommended, win_depth = get_classification()

    st.markdown("""
The game tree shows all positions reachable from a given state up to a chosen depth.
**Green thick edges** are SPE-recommended moves. Node color = classification.
    """)

    col1, col2 = st.columns([2,1])
    with col2:
        depth = st.slider("Tree depth", 1, 4, 3)
        start_options = {
            "(1,1,1,1) — start":  (1,1,1,1),
            "(4,3,1,0) — P1 wins d=1": (4,3,1,0),
            "(2,0,4,0) — P1 wins d=2": (2,0,4,0),
            "(3,3,3,3) — tying":  (3,3,3,3),
            "(1,0,4,3) — losing": (1,0,4,3),
        }
        chosen = st.selectbox("Starting position", list(start_options.keys()))
        start = start_options[chosen]

        cls = classify_state(start, winning, losing)
        st.markdown(f"Classification: {badge(start, winning, losing)}", unsafe_allow_html=True)
        recs = recommended.get(start,[])
        if recs:
            st.markdown(f"SPE move → `{fmt(recs[0])}`")

    with col1:
        with st.spinner("Rendering tree..."):
            fig = fig_game_tree(start, winning, losing, tying, recommended, nexts_map, depth)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("""
<div class="info-callout">
<strong>Reading the tree:</strong>
Each node is labeled <code>(p1_left, p1_right, p2_left, p2_right)</code> with left≥right.
After every move the turn flips and hands are renormalized — so "P1" in the state always means
the player whose turn it is <em>next</em>. A bold green arrow means: this is the move the current
player should take under SPE.
</div>
""", unsafe_allow_html=True)


def page_play():
    st.header("🎮 Play Against the SPE Bot")
    all_states, nexts_map = get_all_states_and_nexts()
    winning, losing, tying, recommended, win_depth = get_classification()

    if "game_state" not in st.session_state:
        st.session_state.game_state = (1,1,1,1)
        st.session_state.history    = [((1,1,1,1), "Game started")]
        st.session_state.human_turn = True   # human is P1
        st.session_state.game_over  = False
        st.session_state.outcome    = ""

    gs    = st.session_state.game_state
    hist  = st.session_state.history
    is_ht = st.session_state.human_turn

    # ── Board display ──
    a, b, c, d = gs
    col_p1, col_mid, col_p2 = st.columns([2,1,2])
    with col_p1:
        st.markdown("### You (P1)")
        st.markdown(f"""
<div class="hand-display">
  {''.join('🖐️' if i < a else '🤚' for i in range(4)) if a>0 else '💀'}
  &nbsp;&nbsp;
  {''.join('🖐️' if i < b else '🤚' for i in range(4)) if b>0 else '💀'}
</div>
<div style="text-align:center;font-family:monospace;font-size:1.2rem;color:#444">
  Left: {a}  &nbsp;|&nbsp;  Right: {b}
</div>
""", unsafe_allow_html=True)

    with col_mid:
        st.markdown("<br><br>", unsafe_allow_html=True)
        cls = classify_state(gs, winning, losing)
        st.markdown(badge(gs, winning, losing), unsafe_allow_html=True)
        st.caption(f"State: `{fmt(gs)}`")

    with col_p2:
        st.markdown("### Bot (P2)")
        st.markdown(f"""
<div class="hand-display">
  {''.join('🖐️' if i < c else '🤚' for i in range(4)) if c>0 else '💀'}
  &nbsp;&nbsp;
  {''.join('🖐️' if i < d else '🤚' for i in range(4)) if d>0 else '💀'}
</div>
<div style="text-align:center;font-family:monospace;font-size:1.2rem;color:#444">
  Left: {c}  &nbsp;|&nbsp;  Right: {d}
</div>
""", unsafe_allow_html=True)

    st.divider()

    if st.session_state.game_over:
        outcome = st.session_state.outcome
        if "P1" in outcome:
            st.success(f"🎉 {outcome} — You win!")
        elif "P2" in outcome:
            st.error(f"🤖 {outcome} — Bot wins!")
        else:
            st.info(f"↺ {outcome} — Cycle detected, declaring draw.")
        if st.button("New Game"):
            for k in ["game_state","history","human_turn","game_over","outcome"]:
                del st.session_state[k]
            st.rerun()
        return

    # ── Human move ──
    if is_ht:
        st.subheader("Your turn — choose a move")
        available = nexts_map.get(gs, [])
        if not available:
            st.session_state.game_over = True
            st.session_state.outcome   = "P2 wins (P1 stuck)"
            st.rerun()

        move_cols = st.columns(min(len(available), 4))
        for i, mv in enumerate(available):
            mc = state_color(mv, winning, losing)
            label = fmt(mv)
            cls_mv = classify_state(mv, winning, losing)
            hint = {"win":"(good for you)","lose":"(bad for you)","tie":"(neutral)"}.get(cls_mv,"")
            with move_cols[i % len(move_cols)]:
                if st.button(f"{label}\n{hint}", key=f"mv_{i}",
                             use_container_width=True):
                    st.session_state.history.append((mv, "You moved"))
                    st.session_state.game_state = mv
                    st.session_state.human_turn = False
                    if mv[0]==0 and mv[1]==0:
                        st.session_state.game_over = True
                        st.session_state.outcome   = "P2 wins"
                    elif mv[2]==0 and mv[3]==0:
                        st.session_state.game_over = True
                        st.session_state.outcome   = "P1 wins"
                    st.rerun()
    else:
        # Bot (SPE) move
        st.subheader("Bot's turn (SPE)...")
        bot_moves = recommended.get(gs, nexts_map.get(gs,[]))
        if not bot_moves:
            st.session_state.game_over = True
            st.session_state.outcome   = "P1 wins (bot stuck)"
            st.rerun()
        bot_mv = bot_moves[0]
        st.markdown(f"Bot plays: `{fmt(gs)}` → **`{fmt(bot_mv)}`**")

        # Detect cycle
        visited_states = [h[0] for h in hist]
        if visited_states.count(bot_mv) >= 3:
            st.session_state.game_over = True
            st.session_state.outcome   = "Draw — infinite cycle detected"
            st.rerun()

        st.session_state.history.append((bot_mv, "Bot moved"))
        st.session_state.game_state = bot_mv
        st.session_state.human_turn = True
        if bot_mv[0]==0 and bot_mv[1]==0:
            st.session_state.game_over = True
            st.session_state.outcome   = "P2 wins"
        elif bot_mv[2]==0 and bot_mv[3]==0:
            st.session_state.game_over = True
            st.session_state.outcome   = "P1 wins"

        if st.button("Continue →"):
            st.rerun()

    st.divider()
    with st.expander("📜 Move History"):
        for i,(s,who) in enumerate(hist):
            cls = classify_state(s, winning, losing)
            color_dot = {"win":"🟢","lose":"🔴","tie":"🟣"}[cls]
            st.markdown(f"{color_dot} Step {i}: `{fmt(s)}` — *{who}*")

    with st.expander("💡 SPE Hint"):
        rec = recommended.get(gs,[])
        if rec:
            st.markdown(f"Optimal move from `{fmt(gs)}`: **`{fmt(rec[0])}`**")
        else:
            st.markdown("No moves available — you've lost.")


def page_table():
    st.header("📋 Full State Table")
    import pandas as pd
    all_states, nexts_map = get_all_states_and_nexts()
    winning, losing, tying, recommended, win_depth = get_classification()

    rows = []
    for s in all_states:
        cls = classify_state(s, winning, losing)
        recs = recommended.get(s,[])
        rows.append({
            "State":      fmt(s),
            "P1 left":    s[0],
            "P1 right":   s[1],
            "P2 left":    s[2],
            "P2 right":   s[3],
            "P1 total":   s[0]+s[1],
            "P2 total":   s[2]+s[3],
            "Type":       cls.upper(),
            "Depth":      win_depth.get(s,""),
            "SPE move":   fmt(recs[0]) if recs else "—",
        })
    df = pd.DataFrame(rows)

    st.markdown("Filter and explore all 200 reachable states.")
    col1,col2,col3 = st.columns(3)
    with col1:
        ftype = st.multiselect("Type", ["WIN","LOSE","TIE"], default=["WIN","LOSE","TIE"])
    with col2:
        fp1 = st.slider("P1 total range", 0, 8, (0,8))
    with col3:
        fp2 = st.slider("P2 total range", 0, 8, (0,8))

    mask = (df["Type"].isin(ftype) &
            df["P1 total"].between(*fp1) &
            df["P2 total"].between(*fp2))
    st.markdown(f"**{mask.sum()} states** match your filters.")
    st.dataframe(df[mask], use_container_width=True, hide_index=True,
                 column_config={
                     "Type": st.column_config.TextColumn("Type"),
                     "Depth": st.column_config.NumberColumn("Depth"),
                 })

    st.download_button("⬇ Download as CSV",
                       df.to_csv(index=False).encode("utf-8"),
                       "chopsticks_spe_states.csv", "text/csv")


# ══════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════

def main():
    page = sidebar()

    if   page == "🏠 Introduction":      page_intro()
    elif page == "🔬 SPE Analysis":       page_analysis()
    elif page == "📊 Visualizations":     page_visualizations()
    elif page == "🌲 Game Tree":          page_game_tree()
    elif page == "🎮 Play the Game":      page_play()
    elif page == "📋 Full State Table":   page_table()

if __name__ == "__main__":
    main()
