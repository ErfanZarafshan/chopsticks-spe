"""
Chopsticks SPE Explorer — Streamlit App (True No-Rollover)
===========================================================
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
import pandas as pd
from collections import defaultdict

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Chopsticks SPE Explorer",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1,h2,h3,h4 { font-family: 'IBM Plex Sans', sans-serif; font-weight:600; }

.hero-box {
    background: linear-gradient(135deg,#0f2027 0%,#1a3a4a 50%,#203a43 100%);
    border-radius:16px; padding:2rem 2.5rem; margin-bottom:1.5rem;
    border:1px solid #2a5a6a;
}
.hero-box h1 { color:#e0f4f1; font-size:2.2rem; margin:0 0 .5rem; }
.hero-box p  { color:#a8d8d0; font-size:1.05rem; margin:0; line-height:1.6; }
.hero-author { margin-top:1rem; font-size:.92rem; color:#7ecfc5;
               border-top:1px solid #2a5a6a; padding-top:.8rem; }

.verdict-win  { background:#e8f8f2; border-left:5px solid #1D9E75; padding:1rem 1.2rem;
                border-radius:8px; margin:.8rem 0; }
.verdict-lose { background:#fff0eb; border-left:5px solid #D85A30; padding:1rem 1.2rem;
                border-radius:8px; margin:.8rem 0; }
.verdict-key  { background:#f0f8ff; border-left:5px solid #185FA5; padding:1rem 1.2rem;
                border-radius:8px; margin:.8rem 0; }

.badge-win  { background:#1D9E75; color:#fff; padding:3px 12px; border-radius:12px;
              font-size:.82rem; font-weight:600; }
.badge-lose { background:#D85A30; color:#fff; padding:3px 12px; border-radius:12px;
              font-size:.82rem; font-weight:600; }

.info-callout { background:#f0f4f8; border-radius:10px; padding:1rem 1.3rem;
                border:1px solid #cdd8e3; margin:.8rem 0; font-size:.93rem;
                line-height:1.65; }
.key-result { background:linear-gradient(135deg,#fff0eb,#ffe4d6);
              border:2px solid #D85A30; border-radius:12px; padding:1.2rem 1.5rem;
              margin:1rem 0; text-align:center; }
.key-result h3 { color:#A32D2D; margin:0 0 .4rem; font-size:1.3rem; }
.key-result p  { color:#5a2020; margin:0; font-size:.95rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 1. GAME ENGINE — TRUE NO-ROLLOVER
# ══════════════════════════════════════════════════════════════

class BadMove(Exception):
    pass

def normalize(pos):
    """True no-rollover: >= 5 -> 0 permanently. Sort left >= right."""
    a, b, c, d = pos
    if a >= 5: a = 0        # no mod-5: hand is dead forever
    if b >= 5: b = 0
    if c >= 5: c = 0
    if d >= 5: d = 0
    if b > a: a, b = b, a
    if d > c: c, d = d, c
    return (a, b, c, d)

def next_turn(pos):
    a, b, c, d = pos
    return (c, d, a, b)

def move_tap(pos, use_left, tap_left):
    a, b, c, d = pos
    src = a if use_left else b
    if src == 0: raise BadMove("Hand is dead")
    if tap_left and c == 0: raise BadMove("Target dead")
    if not tap_left and d == 0: raise BadMove("Target dead")
    if tap_left: c += src
    else: d += src
    return next_turn(normalize((a, b, c, d)))

@st.cache_data
def get_all_states_and_nexts():
    nexts_map = {}
    def compute(pos):
        if pos in nexts_map: return nexts_map[pos]
        a, b, c, d = pos
        if (a==0 and b==0) or (c==0 and d==0):
            nexts_map[pos] = []; return []
        r = set()
        for ul in (True, False):
            for tl in (True, False):
                try: r.add(move_tap(pos, ul, tl))
                except BadMove: pass
        nexts_map[pos] = sorted(r)
        return nexts_map[pos]

    start = (1,1,1,1)
    visited = {start}
    frontier = [start]
    while frontier:
        nxt = []
        for p in frontier:
            for n in compute(p):
                if n not in visited:
                    visited.add(n); nxt.append(n)
        frontier = nxt
    for s in visited: compute(s)
    return sorted(visited), nexts_map

@st.cache_data
def get_classification():
    all_states, nexts_map = get_all_states_and_nexts()

    term = set(s for s in all_states if s[0]==0 and s[1]==0 and (s[2]>0 or s[3]>0))
    losing = set(term); winning = set()

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

    win_depth = {}
    q = list(term)
    for s in term: win_depth[s] = 0
    while q:
        s = q.pop(0)
        for p in all_states:
            if s in nexts_map.get(p,[]) and p not in win_depth:
                win_depth[p] = win_depth[s]+1; q.append(p)

    recommended = {}
    for s in all_states:
        ns = nexts_map[s]
        if s in winning:
            lm = sorted([n for n in ns if n in losing], key=lambda x: win_depth.get(x,999))
            recommended[s] = lm
        else:
            recommended[s] = []

    return winning, losing, recommended, win_depth


# ══════════════════════════════════════════════════════════════
# 2. HELPERS
# ══════════════════════════════════════════════════════════════

WIN_C  = "#1D9E75"
LOSE_C = "#D85A30"
BG_C   = "#FAFAF8"
P_BG   = "#F4F3EE"
EDGE_C = "#888780"

def fmt(pos): return f"({','.join(map(str,pos))})"

def cls(s, winning, losing):
    return "win" if s in winning else "lose"

def badge(s, winning, losing):
    t = cls(s, winning, losing)
    label = "WIN (current player)" if t == "win" else "LOSE (current player)"
    return f'<span class="badge-{t}">{label}</span>'

def sc(s, winning):
    return WIN_C if s in winning else LOSE_C

def simulate(start, winning, losing, recommended, nexts_map, max_steps=40):
    path = [(start, "Start", "P1")]
    pos = start
    for step in range(max_steps):
        mover = "P1" if step % 2 == 0 else "P2"
        moves = recommended.get(pos, []) if pos in winning else nexts_map.get(pos, [])
        if not moves:
            break
        pos = moves[0]
        nm = "P2" if mover == "P1" else "P1"
        path.append((pos, f"{mover} moved", nm))
        # After the move, pos[0,1] = new current player's hands
        # If new current player's hands are dead, the MOVER wins (they just killed them)
        if pos[0] == 0 and pos[1] == 0:
            return path, f"{mover} wins"
        # pos[2,3] dying from opponent's tap is impossible (tapper's hands never change)
    return path, "?"


# ══════════════════════════════════════════════════════════════
# 3. MATPLOTLIB FIGURES
# ══════════════════════════════════════════════════════════════

def fig_overview(winning, losing, win_depth, all_states):
    fig = plt.figure(figsize=(15,8), facecolor=BG_C)
    fig.suptitle(
        "Chopsticks (True No-Rollover) — Complete SPE Analysis\n"
        "(1,1,1,1) = LOSING for P1   →   P2 wins with perfect play   |   0 tying states",
        fontsize=13, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2,3,figure=fig,hspace=.46,wspace=.38,
                           left=.07,right=.97,top=.90,bottom=.09)

    ax0 = fig.add_subplot(gs[0,0])
    _,_,auts = ax0.pie([len(winning),len(losing)],
                       labels=[f"Winning\n{len(winning)}",f"Losing\n{len(losing)}"],
                       colors=[WIN_C,LOSE_C],autopct="%1.0f%%",startangle=90,
                       textprops={"fontsize":10},wedgeprops={"linewidth":1.5,"edgecolor":BG_C})
    for a in auts: a.set(fontsize=9,color="white",fontweight="bold")
    ax0.set_title("92 reachable states\n(zero tying — pure DAG)",fontsize=10,pad=8)

    ax1 = fig.add_subplot(gs[0,1])
    wdv=[win_depth[s] for s in winning if s in win_depth]
    ax1.hist(wdv,bins=range(1,max(wdv)+2),color=WIN_C,edgecolor=BG_C,alpha=.85)
    ax1.set(xlabel="Depth (moves to force win)",ylabel="# states",title="Winning State Depths")
    ax1.set_facecolor(P_BG); ax1.tick_params(labelsize=8)
    ax1.set_xticks(range(1,max(wdv)+1))

    ax2 = fig.add_subplot(gs[0,2])
    ldv=[win_depth[s] for s in losing if s in win_depth]
    if ldv:
        ax2.hist(ldv,bins=range(0,max(ldv)+2),color=LOSE_C,edgecolor=BG_C,alpha=.85)
    ax2.set(xlabel="Depth from terminal loss",ylabel="# states",title="Losing State Depths")
    ax2.set_facecolor(P_BG); ax2.tick_params(labelsize=8)

    ax3 = fig.add_subplot(gs[1,:])
    for s in winning: ax3.scatter(s[0]+s[1],s[2]+s[3],color=WIN_C,alpha=.7,s=45,zorder=3)
    for s in losing:  ax3.scatter(s[0]+s[1],s[2]+s[3],color=LOSE_C,alpha=.85,s=55,zorder=4,marker="X")
    ax3.scatter(2,2,color="gold",s=200,zorder=5,marker="*",edgecolors="#8B6914",linewidth=1.5)
    ax3.set(xlabel="P1 total fingers",ylabel="P2 total fingers",
            title="State Map  (circle=Winning, X=Losing, star=Start (1,1,1,1))")
    ax3.set_facecolor(P_BG); ax3.tick_params(labelsize=9)
    ax3.set_xticks(range(9)); ax3.set_yticks(range(9))
    ax3.legend(handles=[
        mpatches.Patch(color=WIN_C, label=f"Winning ({len(winning)})"),
        mpatches.Patch(color=LOSE_C,label=f"Losing ({len(losing)})"),
    ],fontsize=9,loc="upper right")
    return fig


def fig_heatmap(all_states, winning, losing):
    cell = defaultdict(lambda:{"W":0,"L":0})
    for s in all_states:
        k=(s[0]+s[1],s[2]+s[3])
        if s in winning: cell[k]["W"]+=1
        else: cell[k]["L"]+=1
    fig,axes=plt.subplots(1,2,figsize=(13,5),facecolor=BG_C)
    fig.suptitle("Heatmaps: count of W/L states per (P1 total, P2 total) cell",
                 fontsize=12,fontweight="bold")
    for ax,(title,key,cmap) in zip(axes,[("Winning","W","Greens"),("Losing","L","Reds")]):
        g=np.array([[cell.get((r,c),{}).get(key,0) for c in range(9)] for r in range(9)])
        im=ax.imshow(g,cmap=cmap,origin="lower",aspect="auto",vmin=0,vmax=max(1,g.max()))
        ax.set(title=title,xlabel="P2 total",ylabel="P1 total")
        ax.set_xticks(range(9)); ax.set_yticks(range(9)); ax.tick_params(labelsize=8)
        ax.add_patch(plt.Rectangle((1.5,1.5),1,1,fill=False,edgecolor="gold",lw=2.5))
        for i in range(9):
            for j in range(9):
                v=g[i,j]
                if v>0:
                    ax.text(j,i,str(int(v)),ha="center",va="center",fontsize=9,
                            fontweight="bold",color="white" if v>2 else "#333")
        plt.colorbar(im,ax=ax,shrink=.75)
    axes[0].text(2,2,"START",ha="center",va="center",fontsize=7,color="gold",fontweight="bold")
    plt.tight_layout()
    return fig


def fig_game_tree(start, winning, losing, recommended, nexts_map, depth=4):
    G = nx.DiGraph()
    nc={}; lbl={}; re=set()
    def color(s): return WIN_C if s in winning else LOSE_C
    def bfs(p,d):
        if d==0: return
        G.add_node(p); nc[p]=color(p); lbl[p]=fmt(p)
        for n in nexts_map.get(p,[]):
            G.add_node(n); nc[n]=color(n); lbl[n]=fmt(n)
            G.add_edge(p,n)
            if n in recommended.get(p,[]): re.add((p,n))
            bfs(n,d-1)
    bfs(start,depth)
    fig,ax=plt.subplots(figsize=(16,9),facecolor=BG_C)
    ax.set_facecolor(BG_C)
    ax.set_title(f"SPE Game Tree from {fmt(start)} — depth {depth}\n"
                 "Thick green = SPE-recommended move  |  Green node = Winning  Red = Losing",
                 fontsize=12,pad=10)
    try:    pl=nx.nx_agraph.graphviz_layout(G,prog="dot")
    except: pl=nx.spring_layout(G,seed=42,k=2.2)
    nl=list(G.nodes())
    nx.draw_networkx_nodes(G,pl,nodelist=nl,
                           node_color=[nc.get(n,LOSE_C) for n in nl],
                           node_size=900,ax=ax,alpha=.9)
    nx.draw_networkx_labels(G,pl,labels={n:lbl.get(n,"") for n in nl},
                            font_size=6.5,font_color="white",font_weight="bold",ax=ax)
    reg=[(u,v) for u,v in G.edges() if (u,v) not in re]
    nx.draw_networkx_edges(G,pl,edgelist=reg,edge_color=EDGE_C,alpha=.4,
                           arrows=True,arrowsize=12,connectionstyle="arc3,rad=0.1",ax=ax)
    nx.draw_networkx_edges(G,pl,edgelist=list(re),edge_color=WIN_C,width=2.8,alpha=.9,
                           arrows=True,arrowsize=16,connectionstyle="arc3,rad=0.1",ax=ax)
    ax.legend(handles=[
        mpatches.Patch(color=WIN_C, label="Winning state"),
        mpatches.Patch(color=LOSE_C,label="Losing state"),
    ],fontsize=9,loc="upper right")
    ax.axis("off")
    plt.tight_layout()
    return fig


def fig_path_strip(path, winning, n_show=9):
    path = path[:n_show]
    n = len(path)
    fig, axes = plt.subplots(1, n, figsize=(max(12, n*1.65), 4.0), facecolor=BG_C)
    if n == 1: axes = [axes]
    fig.suptitle(
        "SPE Optimal Game Path from (1,1,1,1) — P2 wins in 8 moves\n"
        "P1 has no escape: every move leads to a Winning state for P2",
        fontsize=11, fontweight="bold", y=1.03
    )
    def s_col(s): return WIN_C if s in winning else LOSE_C
    def s_lbl(s): return "WIN" if s in winning else "LOSE"
    for i, entry in enumerate(path):
        state, who = entry[0], entry[1]   # safe unpack — ignore extra fields
        ax = axes[i]
        a, b, c, d = state
        col = s_col(state)
        ax.set_facecolor(col+"18"); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
        ax.text(.5,.97,f"Step {i}",ha="center",va="top",fontsize=8,color="#555",transform=ax.transAxes)
        ax.text(.5,.87,who,ha="center",va="top",fontsize=7.5,style="italic",color="#555",transform=ax.transAxes)
        ax.text(.5,.75,"P1",ha="center",fontsize=9,fontweight="bold",color="#222",transform=ax.transAxes)
        p1s = "dead" if a+b==0 else f"{'●'*a}{'○'*(4-a)}|{'●'*b}{'○'*(4-b)}"
        ax.text(.5,.63,p1s,ha="center",fontsize=8,color=LOSE_C if a+b==0 else "#222",
                transform=ax.transAxes,family="monospace")
        ax.axhline(.55,color="#ddd",lw=.8)
        ax.text(.5,.48,"P2",ha="center",fontsize=9,fontweight="bold",color="#222",transform=ax.transAxes)
        p2s = "dead" if c+d==0 else f"{'●'*c}{'○'*(4-c)}|{'●'*d}{'○'*(4-d)}"
        ax.text(.5,.36,p2s,ha="center",fontsize=8,color=LOSE_C if c+d==0 else "#222",
                transform=ax.transAxes,family="monospace")
        ax.text(.5,.18,s_lbl(state),ha="center",va="center",fontsize=8,fontweight="bold",
                color="white",transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=.3",facecolor=col,edgecolor="none"))
        ax.text(.5,.05,fmt(state),ha="center",va="bottom",fontsize=6.5,color="#999",transform=ax.transAxes)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# 4. SIDEBAR
# ══════════════════════════════════════════════════════════════

def sidebar():
    with st.sidebar:
        st.markdown("## ✋ Chopsticks SPE")
        st.markdown("""
**Ruleset (true no-rollover)**
- Each player starts with **1 finger** per hand
- Tap one live hand onto an opponent's live hand — they gain your count
- **>= 5 fingers → hand permanently dies** (no modulo, no revival)
- You lose when **both hands are dead**
- No splits or transfers
""")
        st.markdown("""
<div style="background:#fff0eb;border-radius:8px;padding:.7rem 1rem;
    border-left:4px solid #D85A30;font-size:.88rem;margin:.5rem 0">
<strong>Key result:</strong><br>
Starting from (1,1,1,1),<br>
<strong style="color:#A32D2D">Player 2 wins</strong> with perfect play.<br>
P1 is in a Losing state. Zero tying states exist.
</div>
""", unsafe_allow_html=True)
        st.divider()
        page = st.radio("Navigate",
                        ["Introduction",
                         "SPE Analysis",
                         "Visualizations",
                         "Game Tree",
                         "Play the Game",
                         "Full State Table"],
                        label_visibility="collapsed")
        st.divider()
        st.caption("Erfan Zarafshan · LSU Economics · Built with Streamlit")
    return page


# ══════════════════════════════════════════════════════════════
# 5. PAGES
# ══════════════════════════════════════════════════════════════

def page_intro():
    st.markdown("""
<div class="hero-box">
  <h1>✋ Chopsticks — SPE Explorer</h1>
  <p>An interactive deep-dive into the <strong>Subgame Perfect Equilibrium</strong> of Chopsticks
  under <strong>true no-rollover</strong> rules. Hands that reach 5 or more fingers die permanently —
  no wraparound. Who wins with perfect play?</p>
  <p class="hero-author">
    Built by <strong style="color:#e0f4f1">Erfan Zarafshan</strong> &nbsp;&middot;&nbsp;
    PhD Student in Economics &nbsp;&middot;&nbsp; <em>Louisiana State University</em>
  </p>
</div>
""", unsafe_allow_html=True)

    # KEY RESULT callout
    st.markdown("""
<div class="key-result">
  <h3>Main Result: Player 2 Wins</h3>
  <p>The starting position (1,1,1,1) is a <strong>Losing state</strong> for Player 1.
  With perfect SPE play, Player 2 wins in exactly <strong>8 moves</strong>.
  There are <strong>zero tying states</strong> — the game always has a definite winner.</p>
</div>
""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("What is Chopsticks?")
        st.markdown("""
A two-player hand game. Each player starts with **1 finger** raised on each hand.

**On your turn:**
1. Pick one of your *live* hands (> 0 fingers)
2. Tap it against one of the *opponent's* live hands
3. Their tapped hand gains your finger count

**A hand dies** the moment it reaches **5 or more** fingers.
In this true no-rollover version, that hand **stays dead forever** — no mod-5 revival.

**You lose** when both your hands are at 0.

**Example:**
- You: `(3, 1)`, Opponent: `(2, 1)`
- Tap your `3` onto their `2` → `2+3=5` → hand **permanently killed** → `(0, 1)`
        """)

    with c2:
        st.subheader("Why No Tying States?")
        st.markdown("""
Unlike the mod-5 rollover version (where dead hands can revive and create infinite cycles),
**true no-rollover makes the game a Directed Acyclic Graph (DAG)**:

- Finger counts on a hand can **only increase** until it dies
- Dead hands **never come back**
- Therefore the game **cannot revisit a state** — no cycles, no draws

This is why backward induction classifies **every** state as either:
- **Winning** — the current player can force a win, or
- **Losing** — the current player loses regardless of what they do

The standard mod-5 Chopsticks creates tying cycles because hands revive. Remove rollover, and the entire tying region collapses to zero.
        """)

    st.divider()
    st.subheader("At a Glance")
    all_states, nexts_map = get_all_states_and_nexts()
    winning, losing, recommended, win_depth = get_classification()
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Total reachable states", len(all_states))
    m2.metric("Winning states", len(winning), help="Current player wins with perfect play")
    m3.metric("Losing states",  len(losing),  help="Current player loses with perfect opponent")
    m4.metric("Tying states",   0,             help="None — game is a finite DAG")

    st.markdown("""
<div class="verdict-lose">
<strong>P1's only move from (1,1,1,1):</strong>
Tap 1 onto opponent's 1 → state becomes <code>(2,1,1,1)</code> — which is a <strong>Winning</strong> state for P2.
P1 has no other option. From every state P1 can reach, P2 has a path to victory.
The game lasts exactly 8 moves under optimal play.
</div>
""", unsafe_allow_html=True)


def page_analysis():
    st.header("SPE Analysis — Full Breakdown")
    all_states, nexts_map = get_all_states_and_nexts()
    winning, losing, recommended, win_depth = get_classification()

    st.subheader("1 · Why True No-Rollover Eliminates Tying States")
    st.markdown("""
The key difference is what happens when a hand reaches **5 or more fingers**:
    """)

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("""
<div style="background:#fff0eb;border:1.5px solid #D85A30;border-radius:10px;padding:1rem 1.2rem">
  <div style="font-weight:700;color:#A32D2D;font-size:.95rem;margin-bottom:.6rem">
    Mod-5 Rollover (old version)
  </div>
  <div style="font-size:.9rem;color:#5a2020;line-height:1.7">
    When a hand reaches 7 fingers → becomes <strong>2</strong> (7 mod 5).<br>
    The hand is <strong>revived</strong> — it comes back alive.<br><br>
    This creates <strong>cycles</strong> in the game graph: states can repeat,
    so backward induction never fully resolves, leaving
    <strong>100 tying (draw) states</strong>.
  </div>
</div>
""", unsafe_allow_html=True)

    with c_right:
        st.markdown("""
<div style="background:#e8f8f2;border:1.5px solid #1D9E75;border-radius:10px;padding:1rem 1.2rem">
  <div style="font-weight:700;color:#0F6E56;font-size:.95rem;margin-bottom:.6rem">
    True No-Rollover (your setup)
  </div>
  <div style="font-size:.9rem;color:#084a39;line-height:1.7">
    When a hand reaches 5 or more → it goes to <strong>0 permanently</strong>.<br>
    The hand is <strong>dead forever</strong> — no revival possible.<br><br>
    Finger counts can only go <strong>up until death</strong>. The game is a
    <strong>finite DAG</strong> with no cycles →
    <strong>zero tying states</strong>.
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
<div class="info-callout">
<strong>What is a DAG?</strong> A Directed Acyclic Graph — a network of states where
you can never return to a state you've already visited. Because hands can only accumulate
fingers until they die and never revive, the game always moves <em>forward</em> toward
termination. This guarantees every state is classified as either Winning or Losing —
with no draws possible.
</div>
""", unsafe_allow_html=True)

    import pandas as pd
    df_compare = pd.DataFrame({
        "Property":       ["Hand revival",   "Game graph",    "Tying states", "Starting position", "Winner"],
        "Mod-5 Rollover": ["Yes (mod 5)",     "Has cycles",   "100",           "(1,1,1,1) = Draw",  "Neither"],
        "No-Rollover":    ["No (stays dead)", "DAG (no cycles)", "0",           "(1,1,1,1) = Lose",  "Player 2"],
    })
    st.dataframe(df_compare, use_container_width=True, hide_index=True)

    st.subheader("2 · Backward Induction Result")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""
<div class="verdict-lose">
<strong>Losing states (35):</strong> The current player has no winning response.
Every move leads to a Winning position for the opponent.<br><br>
<em>Base case:</em> any state (0,0,c,d) — your hands are both dead.
</div>
""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div class="verdict-win">
<strong>Winning states (57):</strong> The current player has at least one move that
puts the opponent in a Losing state. Win depth ranges from 1 (instant kill) to 4.
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="verdict-key">
<strong>Zero Tying States:</strong> Because the game is a finite DAG with no cycles,
backward induction terminates with <em>every</em> state classified as W or L.
There is no category left over.
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.subheader("3 · The SPE Game Path")
    st.markdown("""
Starting from `(1,1,1,1)`, P1 is in a Losing state. Below is the full game under SPE:
the Losing player always makes their first available move, the Winning player plays optimally.
    """)

    path, outcome = simulate((1,1,1,1), winning, losing, recommended, nexts_map)
    cols = st.columns(len(path))
    for i, entry in enumerate(path):
        state, who = entry[0], entry[1]
        with cols[i]:
            col = sc(state, winning)
            c_name = "WIN" if state in winning else "LOSE"
            st.markdown(f"""
<div style="background:{col}22;border:1px solid {col};border-radius:8px;
    padding:.5rem .4rem;text-align:center;font-size:.8rem">
<div style="font-weight:600;color:{col}">Step {i}</div>
<div style="font-size:.72rem;color:#666;margin:.1rem 0">{who}</div>
<div style="font-family:monospace;font-size:.78rem">{fmt(state)}</div>
<div style="margin-top:.3rem">
  <span style="background:{col};color:white;border-radius:6px;padding:1px 7px;
    font-size:.72rem;font-weight:600">{c_name}</span>
</div>
</div>""", unsafe_allow_html=True)

    st.markdown(f"**Outcome:** {outcome}")
    st.markdown("""
<div class="info-callout">
<strong>Reading the path:</strong> State (a,b,c,d) = (current player's left hand, current player's right hand,
opponent's left, opponent's right). After each move the perspective flips — so "P1" and "P2"
alternate as the "current player" in the state representation.
At Step 8, the current player (P1) has both hands dead → P2 wins.
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.subheader("4 · Winning States by Depth")
    st.markdown("Depth = minimum moves for the winning player to eliminate both opponent hands.")

    by_d = defaultdict(list)
    for s in winning: by_d[win_depth.get(s,-1)].append(s)
    for d in sorted(by_d):
        with st.expander(f"Depth {d}  —  {len(by_d[d])} winning states"):
            rows = []
            for s in sorted(by_d[d]):
                r = recommended.get(s,[])
                rows.append({
                    "State": fmt(s),
                    "P1 hands": f"{s[0]},{s[1]}",
                    "P2 hands": f"{s[2]},{s[3]}",
                    "SPE move ->": fmt(r[0]) if r else "—"
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def page_visualizations():
    st.header("Visualizations")
    all_states, nexts_map = get_all_states_and_nexts()
    winning, losing, recommended, win_depth = get_classification()

    tab1, tab2, tab3 = st.tabs(["Overview", "Heatmaps", "SPE Path Strip"])

    with tab1:
        st.subheader("Full State Classification Overview")
        st.markdown("""
Four panels:
- **Pie** (top-left): 57 winning vs 35 losing — zero tying slice
- **Depth histograms**: winning states cluster at depth 1 and 3; losing states are shallow (depth 0–5)
- **Scatter** (bottom): every state by P1 total vs P2 total fingers.
  The gold star marks the starting position (1,1,1,1), firmly in Losing territory.
        """)
        fig = fig_overview(winning, losing, win_depth, all_states)
        st.pyplot(fig, use_container_width=True); plt.close()

    with tab2:
        st.subheader("Classification Heatmaps")
        st.markdown("""
Each cell `(row = P1 total fingers, col = P2 total fingers)` shows how many W or L states
exist with those totals. The **gold square** marks P1-total=2, P2-total=2 — where the
starting state (1,1,1,1) lives. It appears in the Losing heatmap, confirming P1 loses.
        """)
        fig = fig_heatmap(all_states, winning, losing)
        st.pyplot(fig, use_container_width=True); plt.close()

    with tab3:
        st.subheader("SPE Game Path Strip")
        st.markdown("""
Each card = one step. P1 (Losing from the start) makes their only move.
P2 responds with the SPE move (always landing P1 in a Losing state).
After 8 moves, P1's both hands are dead. Game over.
        """)
        path, outcome = simulate((1,1,1,1), winning, losing, recommended, nexts_map)
        st.markdown(f"**Outcome:** `{outcome}`")
        fig = fig_path_strip(path, winning, n_show=9)
        st.pyplot(fig, use_container_width=True); plt.close()


def page_game_tree():
    st.header("SPE Game Tree")
    all_states, nexts_map = get_all_states_and_nexts()
    winning, losing, recommended, win_depth = get_classification()

    col1, col2 = st.columns([2,1])
    with col2:
        depth = st.slider("Tree depth", 1, 5, 4)
        start_opts = {
            "(1,1,1,1) — start [LOSE]":  (1,1,1,1),
            "(4,3,1,0) — P2 wins d=1":   (4,3,1,0),
            "(2,0,3,2) — P2 wins d=2":   (2,0,3,2),
            "(1,0,1,0) — P2 wins d=3":   (1,0,1,0),
            "(2,1,2,1) — LOSE":          (2,1,2,1),
            "(3,2,2,2) — LOSE":          (3,2,2,2),
        }
        chosen = st.selectbox("Starting position", list(start_opts.keys()))
        start = start_opts[chosen]
        c = "WIN" if start in winning else "LOSE"
        st.markdown(f"Classification: {badge(start, winning, losing)}", unsafe_allow_html=True)
        recs = recommended.get(start,[])
        if recs: st.markdown(f"SPE move → `{fmt(recs[0])}`")
        else: st.markdown("No SPE move — this is a Losing state (all moves lead to W for opponent)")

    with col1:
        with st.spinner("Rendering tree..."):
            fig = fig_game_tree(start, winning, losing, recommended, nexts_map, depth)
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("""
<div class="info-callout">
<strong>Reading the tree:</strong>
<code>(a,b,c,d)</code> = (current player left, current player right, opponent left, opponent right).
Green bold arrows = SPE move. After each move the perspective swaps, so P1/P2 alternate.
In a DAG like this, no state appears twice — the tree is finite and always terminates.
</div>
""", unsafe_allow_html=True)


def _hand_display(n):
    """Return finger display string: filled + empty slots."""
    if n == 0:
        return "DEAD"
    return "●" * n + "○" * (4 - n)

def _hand_color(n):
    return LOSE_C if n == 0 else "#1a1a1a"

def page_play():
    st.header("Play Against the SPE Bot")
    st.markdown("""
You are **Player 1**. The bot is **Player 2** and always plays the SPE strategy.
Because (1,1,1,1) is a **Losing state** for P1, the bot will win with perfect play.
Can you find any deviation that delays it longer than 8 moves?
    """)

    all_states, nexts_map = get_all_states_and_nexts()
    winning, losing, recommended, win_depth = get_classification()

    # ── Init session state ──────────────────────────────────────
    if "game_state" not in st.session_state:
        st.session_state.game_state  = (1, 1, 1, 1)
        st.session_state.history     = [((1, 1, 1, 1), "Game started")]
        st.session_state.human_turn  = True
        st.session_state.game_over   = False
        st.session_state.outcome     = ""
        st.session_state.move_count  = 0

    gs    = st.session_state.game_state
    is_ht = st.session_state.human_turn

    # ── Resolve ABSOLUTE P1 / P2 hands regardless of perspective ─
    # State is always (current_player_L, current_player_R, opponent_L, opponent_R)
    # When human_turn=True  → positions 0,1 = human (P1), 2,3 = bot (P2)
    # When human_turn=False → positions 0,1 = bot  (P2), 2,3 = human (P1)
    if is_ht:
        p1a, p1b = gs[0], gs[1]   # human's hands
        p2a, p2b = gs[2], gs[3]   # bot's hands
    else:
        p1a, p1b = gs[2], gs[3]   # human's hands (opponent in state)
        p2a, p2b = gs[0], gs[1]   # bot's hands   (current player in state)

    # ── Board ───────────────────────────────────────────────────
    col_p1, col_mid, col_p2 = st.columns([5, 3, 5])

    with col_p1:
        p1_cls = "LOSE" if (p1a == 0 and p1b == 0) else ("WIN" if gs in winning and is_ht else "LOSE" if gs in losing and is_ht else "")
        st.markdown(f"""
<div style="background:#f8f8f6;border:1.5px solid #ddd;border-radius:12px;padding:1rem;text-align:center">
  <div style="font-weight:600;font-size:1rem;color:#333;margin-bottom:.5rem">You — Player 1</div>
  <div style="font-family:monospace;font-size:1.6rem;letter-spacing:4px;margin:.3rem 0">
    <span style="color:{_hand_color(p1a)}">{_hand_display(p1a)}</span>
    &nbsp;&nbsp;
    <span style="color:{_hand_color(p1b)}">{_hand_display(p1b)}</span>
  </div>
  <div style="font-size:.85rem;color:#666;margin-top:.3rem">Left: {p1a} &nbsp;|&nbsp; Right: {p1b}</div>
</div>""", unsafe_allow_html=True)

    with col_mid:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        # Show state classification from HUMAN's perspective
        human_cls = cls(gs, winning, losing) if is_ht else ("lose" if gs in winning else "win")
        label = "Your turn" if is_ht else "Bot's turn"
        badge_cls = "win" if human_cls == "win" else "lose"
        badge_label = "WIN" if human_cls == "win" else "LOSE"
        st.markdown(f"""
<div style="text-align:center">
  <div style="font-size:.78rem;color:#888;margin-bottom:.3rem">{label}</div>
  <span class="badge-{badge_cls}">{badge_label} for you</span>
  <div style="font-size:.72rem;color:#999;margin-top:.4rem;font-family:monospace">{fmt(gs)}</div>
  <div style="font-size:.72rem;color:#aaa">Move {st.session_state.move_count}</div>
</div>""", unsafe_allow_html=True)

    with col_p2:
        st.markdown(f"""
<div style="background:#f8f8f6;border:1.5px solid #ddd;border-radius:12px;padding:1rem;text-align:center">
  <div style="font-weight:600;font-size:1rem;color:#333;margin-bottom:.5rem">Bot — Player 2 (SPE)</div>
  <div style="font-family:monospace;font-size:1.6rem;letter-spacing:4px;margin:.3rem 0">
    <span style="color:{_hand_color(p2a)}">{_hand_display(p2a)}</span>
    &nbsp;&nbsp;
    <span style="color:{_hand_color(p2b)}">{_hand_display(p2b)}</span>
  </div>
  <div style="font-size:.85rem;color:#666;margin-top:.3rem">Left: {p2a} &nbsp;|&nbsp; Right: {p2b}</div>
</div>""", unsafe_allow_html=True)

    st.divider()

    # ── Game over ────────────────────────────────────────────────
    if st.session_state.game_over:
        outcome = st.session_state.outcome
        n_moves = st.session_state.move_count
        if "P1" in outcome:
            st.success(f"You win in {n_moves} moves! {outcome} (rare — bot made an error or you got lucky)")
        elif "P2" in outcome:
            st.error(f"Bot wins in {n_moves} moves! The SPE predicts exactly 8 — you {'matched' if n_moves==8 else 'deviated from'} the optimal line.")
        else:
            st.info(outcome)
        if st.button("New Game", type="primary"):
            for k in ["game_state","history","human_turn","game_over","outcome","move_count"]:
                if k in st.session_state: del st.session_state[k]
            st.rerun()
        return

    # ── Human turn ────────────────────────────────────────────────
    if is_ht:
        st.subheader("Your turn — tap a hand")
        available = nexts_map.get(gs, [])
        if not available:
            st.session_state.game_over = True
            st.session_state.outcome   = "P2 wins — P1 has no moves left"
            st.rerun()
            return

        n_cols = min(len(available), 4)
        move_cols = st.columns(n_cols)
        for i, mv in enumerate(available):
            mv_human_cls = "lose" if mv in winning else "win"   # from human's next perspective
            mv_label = "Good for bot" if mv in winning else "Bad for bot"
            border_col = LOSE_C if mv in winning else WIN_C
            with move_cols[i % n_cols]:
                if st.button(
                    f"{fmt(mv)}\n{mv_label}",
                    key=f"mv_{i}",
                    use_container_width=True
                ):
                    st.session_state.history.append((mv, "You moved"))
                    st.session_state.game_state  = mv
                    st.session_state.human_turn  = False
                    st.session_state.move_count += 1
                    # Check win/lose after human move
                    # After human moves, new state's positions 0,1 = bot, 2,3 = human
                    if mv[2] == 0 and mv[3] == 0:   # human's hands now at 0 (opponent in new state)
                        st.session_state.game_over = True
                        st.session_state.outcome   = "P2 wins"
                    elif mv[0] == 0 and mv[1] == 0:  # bot's hands dead
                        st.session_state.game_over = True
                        st.session_state.outcome   = "P1 wins"
                    st.rerun()

    # ── Bot turn — auto plays, no button needed ───────────────────
    else:
        bot_moves = recommended.get(gs, nexts_map.get(gs, []))
        if not bot_moves:
            st.session_state.game_over = True
            st.session_state.outcome   = "P1 wins — bot has no moves"
            st.rerun()
            return

        bot_mv = bot_moves[0]

        # Append exactly once per bot turn (guard against double-append)
        last_entry = st.session_state.history[-1] if st.session_state.history else None
        already_appended = (last_entry is not None and
                            isinstance(last_entry, tuple) and
                            len(last_entry) == 2 and
                            last_entry[0] == bot_mv and
                            last_entry[1] == "Bot moved")
        if not already_appended:
            st.session_state.history.append((bot_mv, "Bot moved"))
            st.session_state.move_count += 1

        st.session_state.game_state  = bot_mv
        st.session_state.human_turn  = True

        # After bot moves: new state positions 0,1 = human, 2,3 = bot
        if bot_mv[2] == 0 and bot_mv[3] == 0:   # bot's hands dead (opponent in new state)
            st.session_state.game_over = True
            st.session_state.outcome   = "P1 wins"
        elif bot_mv[0] == 0 and bot_mv[1] == 0:  # human's hands dead
            st.session_state.game_over = True
            st.session_state.outcome   = "P2 wins"

        st.rerun()   # auto-advance — no button needed

    # ── Expanders ─────────────────────────────────────────────────
    st.divider()

    with st.expander("Move History"):
        hist = st.session_state.history
        if not hist:
            st.caption("No moves yet.")
        else:
            for i, entry in enumerate(hist):
                # Defensive unpack: handle both (state, label) and any legacy format
                try:
                    s, who = entry[0], entry[1]
                    # s should be a 4-tuple; if it's somehow a bare int, skip
                    if not (isinstance(s, tuple) and len(s) == 4):
                        st.caption(f"Step {i}: [invalid entry]")
                        continue
                    dot_color = WIN_C if s in winning else LOSE_C
                    label = "W" if s in winning else "L"
                    st.markdown(
                        f'<span style="color:{dot_color};font-weight:700">[{label}]</span> '
                        f'Step {i}: <code>{fmt(s)}</code> — <em>{who}</em>',
                        unsafe_allow_html=True
                    )
                except Exception:
                    st.caption(f"Step {i}: [could not display entry]")

    with st.expander("SPE Hint"):
        recs = recommended.get(gs, [])
        if recs:
            st.markdown(f"Optimal SPE move from `{fmt(gs)}` → **`{fmt(recs[0])}`**")
            mv_depth = win_depth.get(recs[0], "?")
            st.caption(f"That move puts the opponent in a Losing state at depth {mv_depth}.")
        else:
            st.markdown(
                f"`{fmt(gs)}` is a **Losing state** — every available move hands the bot "
                "a Winning position. There is no escape under perfect play."
            )


def page_table():
    st.header("Full State Table")
    all_states, nexts_map = get_all_states_and_nexts()
    winning, losing, recommended, win_depth = get_classification()

    rows = []
    for s in all_states:
        recs = recommended.get(s,[])
        rows.append({
            "State":       fmt(s),
            "P1 left":     s[0], "P1 right": s[1],
            "P2 left":     s[2], "P2 right": s[3],
            "P1 total":    s[0]+s[1], "P2 total": s[2]+s[3],
            "Type":        "WIN" if s in winning else "LOSE",
            "Depth":       win_depth.get(s,""),
            "SPE move":    fmt(recs[0]) if recs else "— (losing)",
        })
    df = pd.DataFrame(rows)

    c1,c2,c3 = st.columns(3)
    with c1: ftype = st.multiselect("Type", ["WIN","LOSE"], default=["WIN","LOSE"])
    with c2: fp1 = st.slider("P1 total", 0, 8, (0,8))
    with c3: fp2 = st.slider("P2 total", 0, 8, (0,8))

    mask = (df["Type"].isin(ftype) &
            df["P1 total"].between(*fp1) &
            df["P2 total"].between(*fp2))
    st.markdown(f"**{mask.sum()} states** match.")
    st.dataframe(df[mask], use_container_width=True, hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False).encode(),
                       "chopsticks_spe_no_rollover.csv", "text/csv")


# ══════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════

def main():
    page = sidebar()
    if   page == "Introduction":   page_intro()
    elif page == "SPE Analysis":   page_analysis()
    elif page == "Visualizations": page_visualizations()
    elif page == "Game Tree":      page_game_tree()
    elif page == "Play the Game":  page_play()
    elif page == "Full State Table": page_table()

if __name__ == "__main__":
    main()
