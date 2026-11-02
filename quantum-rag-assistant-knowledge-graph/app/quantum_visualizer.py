"""Deterministic, locally generated scientific diagrams for quantum tutoring."""
from __future__ import annotations
import io
import math
import re
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VISUALIZATION_REGISTRY = {
 "shor_circuit":{"title":"Shor's algorithm - conceptual circuit","patterns":("shor",)},
 "grover_circuit":{"title":"Grover's search algorithm","patterns":("grover",)},
 "deutsch_jozsa_circuit":{"title":"Deutsch-Jozsa algorithm","patterns":("deutsch-jozsa","deutsch jozsa")},
 "bernstein_vazirani_circuit":{"title":"Bernstein-Vazirani algorithm","patterns":("bernstein-vazirani","bernstein vazirani")},
 "simon_circuit":{"title":"Simon's algorithm","patterns":("simon's algorithm","simon algorithm")},
 "qft_circuit":{"title":"Quantum Fourier Transform circuit","patterns":("quantum fourier transform","qft")},
 "qpe_circuit":{"title":"Quantum phase estimation","patterns":("quantum phase estimation","phase estimation")},
 "superdense_circuit":{"title":"Superdense coding circuit","patterns":("superdense coding",)},
 "vqe_circuit":{"title":"VQE parameterized circuit (illustrative)","patterns":("variational quantum eigensolver","vqe")},
 "qaoa_circuit":{"title":"QAOA parameterized circuit (illustrative)","patterns":("qaoa","quantum approximate optimization")},
}
SUPPORTED = {"bloch_sphere", "hadamard_gate", "pauli_x_gate", "pauli_y_gate", "pauli_z_gate", "s_gate", "t_gate", "rx_gate", "ry_gate", "rz_gate", "cz_gate", "cnot_gate", "swap_gate", "toffoli_gate", "bell_circuit", "ghz_circuit", "teleportation_circuit", "superposition", "measurement", "interference", *VISUALIZATION_REGISTRY.keys()}


def _normalize(text: str) -> str:
    return (text or "").lower().replace("‑", "-").replace("–", "-").replace("’", "'")


def select_visualizations(question: str, answer: str = "", retrieved_context: str = "") -> dict:
    """Intent-first selection; supporting answer terms never override explicit user intent."""
    q, a = _normalize(question), _normalize(answer)
    rules = [
      ("shor_circuit",("shor",)),("grover_circuit",("grover",)),
      ("deutsch_jozsa_circuit",("deutsch-jozsa","deutsch jozsa")),
      ("bernstein_vazirani_circuit",("bernstein-vazirani","bernstein vazirani")),
      ("simon_circuit",("simon's algorithm","simon algorithm")),
      ("qft_circuit",("quantum fourier transform","qft")),
      ("qpe_circuit",("quantum phase estimation","phase estimation")),
      ("teleportation_circuit",("quantum teleportation","teleportation circuit")),
      ("superdense_circuit",("superdense coding",)),
      ("vqe_circuit",("variational quantum eigensolver","vqe")),
      ("qaoa_circuit",("qaoa","quantum approximate optimization")),
      ("toffoli_gate",("toffoli","ccnot","controlled-controlled-not")),
      ("cnot_gate",("cnot","controlled-not","controlled not","cx gate")),
      ("swap_gate",("swap gate","swap operation")),
      ("hadamard_gate",("hadamard gate","hadamard","h gate")),
      ("pauli_x_gate",("pauli-x","pauli x")),("pauli_y_gate",("pauli-y","pauli y")),
      ("pauli_z_gate",("pauli-z","pauli z")),
      ("s_gate",("s gate","phase gate")),("t_gate",("t gate",)),
      ("rx_gate",("rx gate","rotation x gate")),("ry_gate",("ry gate","rotation y gate")),
      ("rz_gate",("rz gate","rotation z gate")),("cz_gate",("cz gate","controlled-z","controlled z")),
      ("bloch_sphere",("bloch sphere","bloch-sphere")),
      ("ghz_circuit",("ghz state","ghz circuit","greenberger")),
      ("bell_circuit",("bell state","bell pair","bell circuit")),
      ("interference",("quantum interference","constructive interference","destructive interference")),
      ("measurement",("quantum measurement","measurement probabilities")),
      ("superposition",("quantum superposition","superposition state","superposition of")),
    ]
    def match(value):
        padded = " " + value + " "
        for kind, terms in rules:
            if any(term in padded for term in terms): return kind
        return None
    primary = match(q)
    # Only use answer/retrieval context when the user used a genuinely referential
    # follow-up (e.g. "explain it"), never when they named an unsupported subject.
    vague_followup = bool(re.fullmatch(r"\s*(?:explain|describe|show|visualize|diagram|what is|how does)\s+(?:it|this|that|the concept|the process)(?:\s+with.*)?[?.!]*\s*", q))
    if not primary and vague_followup:
        primary = match(a) or match(_normalize(retrieved_context))
    if not primary:
        return {"primary_concept":None,"diagram_type":None,"visualization_type":None,
                "secondary_visualizations":[],"confidence":0.0,"reason":"No supported concept detected."}
    return {"primary_concept":primary,
            "diagram_type":"quantum_circuit" if primary.endswith(("_circuit","_gate")) else "scientific_visualization",
            "visualization_type":primary,"secondary_visualizations":[],
            "confidence":0.96 if match(q) else 0.72,
            "reason":"Selected from the user's question." if match(q) else "Resolved from answer context.",
            "title":VISUALIZATION_REGISTRY.get(primary,{}).get("title",primary.replace("_"," ").title())}


def detect_diagram(question: str, answer: str = "") -> Optional[str]:
    return select_visualizations(question, answer)["visualization_type"]


def _style(ax, title: str):
    ax.set_facecolor("#0b1220")
    ax.set_title(title, color="#e2e8f0", fontsize=15, pad=16, weight="bold")
    ax.tick_params(colors="#cbd5e1")


def _bloch(ax):
    u = np.linspace(0, 2*np.pi, 64)
    v = np.linspace(0, np.pi, 32)
    x = np.outer(np.cos(u), np.sin(v)); y = np.outer(np.sin(u), np.sin(v)); z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="#38bdf8", alpha=.10, linewidth=0, shade=False)
    for vals in (x, y, z):
        pass
    ax.plot([-1.15,1.15],[0,0],[0,0], color="#f87171", lw=1.5)
    ax.plot([0,0],[-1.15,1.15],[0,0], color="#4ade80", lw=1.5)
    ax.plot([0,0],[0,0],[-1.15,1.15], color="#60a5fa", lw=1.5)
    theta, phi = math.pi/3, math.pi/4
    px = math.sin(theta)*math.cos(phi); py=math.sin(theta)*math.sin(phi); pz=math.cos(theta)
    ax.quiver(0,0,0,px,py,pz,color="#fbbf24",linewidth=3,arrow_length_ratio=.12)
    ax.scatter([0,0],[0,0],[1,-1],color=["#4ade80","#fb7185"],s=45)
    ax.text(0,0,1.13,"|0⟩",color="#4ade80",ha="center",fontsize=11)
    ax.text(0,0,-1.18,"|1⟩",color="#fb7185",ha="center",fontsize=11)
    ax.text(1.18,0,0,"X",color="#f87171"); ax.text(0,1.18,0,"Y",color="#4ade80"); ax.text(0,0,1.22,"Z",color="#60a5fa")
    ax.text(px*.55,py*.55,pz*.55,"|ψ⟩",color="#fbbf24",fontsize=12)
    ax.set(xlim=(-1.25,1.25),ylim=(-1.25,1.25),zlim=(-1.25,1.25)); ax.set_box_aspect((1,1,1)); ax.view_init(elev=22,azim=35)
    ax.set_axis_off()


def _circuit(ax, rows, title):
    n=len(rows); ax.set_xlim(-.3,6.4); ax.set_ylim(-1.25,n-.3); ax.axis("off"); _style(ax,title)
    for i, gates in enumerate(rows):
        y=n-1-i; ax.plot([0,6],[y,y],color="#94a3b8",lw=1.5); ax.text(-.2,y,f"q{i}",color="#e2e8f0",ha="right",va="center")
        for x,label in gates:
            if label == "●": ax.scatter([x],[y],s=65,color="#38bdf8",zorder=3)
            elif label == "⊕":
                circ=plt.Circle((x,y),.16,fill=False,color="#38bdf8",lw=1.8); ax.add_patch(circ); ax.text(x,y,label,color="#38bdf8",ha="center",va="center",fontsize=13)
            else:
                rect=plt.Rectangle((x-.19,y-.17),.38,.34,facecolor="#172554",edgecolor="#38bdf8",lw=1.3); ax.add_patch(rect); ax.text(x,y,label,color="#e0f2fe",ha="center",va="center",fontsize=10)
    # Vertical controlled-NOT links
    if n >= 2:
        for x in sorted(set(x for row in rows for x,l in row if l in {"●","⊕"})):
            ys=[n-1-i for i,row in enumerate(rows) for xx,l in row if xx==x and l in {"●","⊕"}]
            if len(ys)>1: ax.plot([x,x],[min(ys),max(ys)],color="#38bdf8",lw=1.3,zorder=1)


def _qiskit_circuit_png(kind: str):
    """Draw real Qiskit circuits when the optional Qiskit extra is installed."""
    try:
        from qiskit import QuantumCircuit
        if kind == "bell_circuit":
            qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1); qc.measure_all()
        elif kind == "ghz_circuit":
            qc = QuantumCircuit(3); qc.h(0); qc.cx(0, 1); qc.cx(0, 2); qc.measure_all()
        elif kind == "teleportation_circuit":
            qc = QuantumCircuit(3, 2)
            qc.h(1); qc.cx(1, 2); qc.cx(0, 1); qc.h(0); qc.measure(0, 0); qc.measure(1, 1)
            # Classical feed-forward corrections are represented as labeled conditional operations.
            with qc.if_test((qc.clbits[1], 1)):
                qc.z(2)
            with qc.if_test((qc.clbits[0], 1)):
                qc.x(2)
        else:
            return None
        figure = qc.draw(output="mpl", style={"backgroundcolor": "#0b1220"})
        out = io.BytesIO(); figure.savefig(out, format="png", dpi=150, bbox_inches="tight", facecolor="#0b1220")
        plt.close(figure); return out.getvalue()
    except (ImportError, ModuleNotFoundError, Exception):
        return None


def render_diagram(kind: str) -> bytes:
    if kind not in SUPPORTED: raise ValueError("Unsupported diagram type")
    qiskit_image = _qiskit_circuit_png(kind) if kind in {"bell_circuit", "ghz_circuit", "teleportation_circuit"} else None
    if qiskit_image: return qiskit_image
    fig = plt.figure(figsize=(8.4,5.2), facecolor="#0b1220")
    if kind == "bloch_sphere":
        ax=fig.add_subplot(111,projection="3d"); _style(ax,"Bloch sphere — single-qubit state"); _bloch(ax)
        fig.text(.5,.04,"A pure qubit state is represented by a point on the sphere surface.",ha="center",color="#cbd5e1",fontsize=9)
    elif kind in {"hadamard_gate", "pauli_x_gate", "pauli_y_gate", "pauli_z_gate", "s_gate", "t_gate", "rx_gate", "ry_gate", "rz_gate", "cz_gate", "cnot_gate", "swap_gate", "toffoli_gate"}:
        ax=fig.add_subplot(111)
        labels = {"hadamard_gate":"H", "pauli_x_gate":"X", "pauli_y_gate":"Y", "pauli_z_gate":"Z", "s_gate":"S", "t_gate":"T", "rx_gate":"RX(θ)", "ry_gate":"RY(θ)", "rz_gate":"RZ(θ)", "cz_gate":"CZ", "cnot_gate":"CNOT", "swap_gate":"SWAP", "toffoli_gate":"CCX"}
        gate = labels[kind]
        if kind == "cz_gate":
            _circuit(ax,[[(2,"●")],[(2,"Z")]],"Controlled-Z (CZ) gate")
            ax.plot([2,2],[0,1],color="#38bdf8",lw=1.3,zorder=1)
            ax.text(3.1,.5,"Applies a phase flip to |11⟩.",color="#cbd5e1",fontsize=9)
        elif kind == "cnot_gate":
            _circuit(ax,[[(2,"●")],[(2,"⊕")]],"Controlled-NOT (CNOT) gate")
            ax.text(3.1,.5,"Control qubit flips the target when control = |1⟩.",color="#cbd5e1",fontsize=9)
        elif kind == "toffoli_gate":
            _circuit(ax,[[(2,"●")],[(2,"●")],[(2,"⊕")]],"Toffoli (CCNOT) gate")
            ax.text(3.1,.5,"Target flips only when both control qubits are |1⟩.",color="#cbd5e1",fontsize=9)
        elif kind == "swap_gate":
            _circuit(ax,[[(2,"×")],[(2,"×")]],"SWAP gate")
            ax.plot([2,2],[0,1],color="#38bdf8",lw=1.5)
        else:
            ax.set_xlim(0,6); ax.set_ylim(-1,1); ax.axis("off"); _style(ax,f"{gate} single-qubit gate")
            ax.plot([.5,5.5],[0,0],color="#94a3b8",lw=2)
            ax.text(.45,0,"|ψ⟩",color="#e2e8f0",ha="right",va="center",fontsize=12)
            rect=plt.Rectangle((2.65,-.32),.7,.64,facecolor="#172554",edgecolor="#38bdf8",lw=2)
            ax.add_patch(rect); ax.text(3,0,gate,color="#e0f2fe",ha="center",va="center",fontsize=17,weight="bold")
            ax.text(5.55,0,"U|ψ⟩",color="#e2e8f0",ha="left",va="center",fontsize=12)
            descriptions = {"hadamard_gate":"H|0⟩ = |+⟩ = (|0⟩ + |1⟩)/√2", "pauli_x_gate":"X flips |0⟩ ↔ |1⟩", "pauli_y_gate":"Y rotates the Bloch vector about the Y axis", "pauli_z_gate":"Z adds a phase of −1 to |1⟩", "s_gate":"S adds a phase of i to |1⟩", "t_gate":"T adds a phase of e^(iπ/4) to |1⟩", "rx_gate":"Rotation by θ around the X axis", "ry_gate":"Rotation by θ around the Y axis", "rz_gate":"Rotation by θ around the Z axis"}
            ax.text(3,-.68,descriptions[kind],color="#cbd5e1",ha="center",fontsize=11)
    elif kind == "bell_circuit":
        ax=fig.add_subplot(111); _circuit(ax,[[(1,"H"),(2,"●")],[(2,"⊕")]],"Bell-state preparation circuit")
        ax.text(3.1,.5,"H on q0 creates superposition; CNOT entangles q0 and q1.",color="#cbd5e1",fontsize=9)
    elif kind == "ghz_circuit":
        ax=fig.add_subplot(111); _circuit(ax,[[(1,"H"),(2,"●"),(3,"●")],[(2,"⊕")],[(3,"⊕")]],"Three-qubit GHZ-state circuit")
    elif kind == "teleportation_circuit":
        ax=fig.add_subplot(111); _circuit(ax,[[(1,"ψ"),(2,"●"),(3,"H"),(4,"M")],[(2,"H"),(3,"⊕"),(4,"M")],[(2,"⊕"),(3,"●"),(5,"Z"),(5.7,"X")]],"Quantum teleportation — conceptual circuit")
    elif kind == "superposition":
        ax=fig.add_subplot(111); _style(ax,"Qubit superposition and measurement probabilities")
        ax.bar([0,1],[.5,.5],color=["#38bdf8","#a78bfa"],width=.55); ax.set_xticks([0,1],["|0⟩","|1⟩"],color="#e2e8f0"); ax.set_ylabel("Probability",color="#cbd5e1"); ax.set_ylim(0,1); ax.grid(axis="y",alpha=.2)
        for s in ax.spines.values(): s.set_color("#475569")
        ax.text(.5,.78,"|+⟩ = (|0⟩ + |1⟩) / √2",ha="center",color="#f8fafc",fontsize=13)
    elif kind == "measurement":
        ax=fig.add_subplot(111); _style(ax,"Measurement probabilities for a sample qubit")
        ax.bar([0,1],[.75,.25],color=["#38bdf8","#a78bfa"],width=.55); ax.set_xticks([0,1],["Outcome 0","Outcome 1"],color="#e2e8f0"); ax.set_ylabel("Probability",color="#cbd5e1"); ax.set_ylim(0,1); ax.grid(axis="y",alpha=.2)
        for s in ax.spines.values(): s.set_color("#475569")
        ax.text(0,.78,"75%",ha="center",color="white"); ax.text(1,.28,"25%",ha="center",color="white")
    elif kind in VISUALIZATION_REGISTRY:
        ax=fig.add_subplot(111)
        layouts = {
          "shor_circuit": ([[(1,"H"),(2,"H"),(3,"●"),(4,"QFT⁻¹"),(5,"M")],[(3,"U_f"),(5,"M")]], "Control register: Hadamards → controlled modular exponentiation; work register: |1⟩; then inverse QFT and measurement"),
          "grover_circuit": ([[(1,"H"),(2,"Oracle"),(3,"Diffusion"),(4,"M")],[(1,"H"),(2,"Oracle"),(3,"Diffusion"),(4,"M")]], "Repeat oracle + diffusion approximately √N times"),
          "deutsch_jozsa_circuit": ([[(1,"H"),(2,"U_f"),(3,"H"),(4,"M")],[(1,"X"),(2,"U_f"),(3,"H")]], "Input register and ancilla; measurement distinguishes constant from balanced"),
          "qft_circuit": ([[(1,"H"),(2,"CP"),(3,"CP"),(4,"SWAP")],[(2,"H"),(3,"CP"),(4,"SWAP")],[(3,"H"),(4,"SWAP")]], "Hadamards + controlled phase rotations + bit-reversal swaps"),
          "qpe_circuit": ([[(1,"H"),(2,"U²⁰"),(3,"QFT⁻¹"),(4,"M")],[(1,"H"),(2,"U²¹"),(3,"QFT⁻¹"),(4,"M")]], "Controlled powers of U; inverse QFT estimates the eigenphase"),
          "vqe_circuit": ([[(1,"RY(θ)"),(2,"RZ(φ)"),(3,"Entangle"),(4,"Measure")],[(1,"RY(θ)"),(2,"RZ(φ)"),(3,"Entangle"),(4,"Measure")]], "Parameterized ansatz → measurements → classical optimizer"),
          "qaoa_circuit": ([[(1,"H"),(2,"U_C(γ)"),(3,"U_B(β)"),(4,"Measure")],[(1,"H"),(2,"U_C(γ)"),(3,"U_B(β)"),(4,"Measure")]], "Alternating cost and mixer layers with variational parameters"),
          "bernstein_vazirani_circuit": ([[(1,"H"),(2,"U_f"),(3,"H"),(4,"M")],[(1,"H"),(2,"U_f")]], "Oracle encodes the hidden bit string; final measurement reveals it"),
          "simon_circuit": ([[(1,"H"),(2,"U_f"),(3,"H"),(4,"M")],[(1,"H"),(2,"U_f"),(4,"M")]], "Repeated samples yield linear constraints used to recover the secret string"),
          "superdense_circuit": ([[(1,"H"),(2,"●"),(3,"Encode"),(4,"Decode")],[(2,"⊕"),(3,"Encode"),(4,"Decode")]], "One transmitted qubit carries two classical bits using a shared Bell pair"),
        }
        rows, subtitle = layouts[kind]
        _circuit(ax, rows, VISUALIZATION_REGISTRY[kind]["title"])
        if kind == "shor_circuit":
            ax.plot([3,3],[0,1],color="#38bdf8",lw=1.4,zorder=1)
            ax.text(-.22,1,"Control",color="#94a3b8",ha="right",va="center",fontsize=7)
            ax.text(-.22,0,"Work",color="#94a3b8",ha="right",va="center",fontsize=7)
        ax.text(3,-.48,subtitle,color="#cbd5e1",ha="center",va="top",fontsize=8,wrap=True)
        if kind == "shor_circuit":
            ax.text(3,-.72,"Classical continued-fraction post-processing estimates period r and verifies candidate factors.",color="#fbbf24",ha="center",fontsize=8)
            ax.text(3,-.94,"Conceptual schematic: U_f is not expanded as executable modular exponentiation.",color="#fca5a5",ha="center",fontsize=7)
    else:
        ax=fig.add_subplot(111); _style(ax,"Quantum interference — phase-dependent amplitudes")
        x=np.linspace(0,2*np.pi,400); ax.plot(x,np.cos(x)**2,label="Constructive phase",color="#38bdf8",lw=2.5); ax.plot(x,np.sin(x)**2,label="Complementary phase",color="#a78bfa",lw=2.5)
        ax.set_xlabel("Relative phase (radians)",color="#cbd5e1"); ax.set_ylabel("Probability",color="#cbd5e1"); ax.set_ylim(0,1); ax.grid(alpha=.2); ax.legend(facecolor="#172033",labelcolor="#e2e8f0")
        for s in ax.spines.values(): s.set_color("#475569")
    fig.tight_layout(pad=2)
    out=io.BytesIO(); fig.savefig(out,format="png",dpi=150,facecolor=fig.get_facecolor(),bbox_inches="tight"); plt.close(fig)
    return out.getvalue()
