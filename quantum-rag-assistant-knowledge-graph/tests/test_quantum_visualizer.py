from app.quantum_visualizer import detect_diagram, render_diagram, SUPPORTED


def test_bloch_sphere_detection():
    assert detect_diagram("Explain Bloch sphere") == "bloch_sphere"


def test_specific_concept_detection():
    assert detect_diagram("Explain quantum teleportation circuit") == "teleportation_circuit"


def test_no_diagram_for_unrelated_question():
    assert detect_diagram("What is the history of quantum computing?") is None


def test_all_diagrams_render_png():
    for kind in SUPPORTED:
        image = render_diagram(kind)
        assert image.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(image) > 5000


def test_auto_visualization_endpoint_returns_inline_png():
    from app.routes.v3 import DiagramRequest, auto_visualization
    response = auto_visualization(DiagramRequest(question="Explain the Bloch sphere", answer="A qubit state on a sphere."))
    assert response.media_type == "image/png"
    assert response.body.startswith(b"\x89PNG\r\n\x1a\n")
    assert response.headers["X-Diagram-Type"] == "bloch_sphere"


def test_auto_visualization_endpoint_skips_unrelated_questions():
    from app.routes.v3 import DiagramRequest, auto_visualization
    result = auto_visualization(DiagramRequest(question="Tell me about quantum history"))
    assert result == {"generated": False, "reason": "No supported visual concept detected."}


def test_hadamard_question_wins_over_bloch_sphere_in_answer():
    assert detect_diagram(
        "Explain Hadamard gate",
        "The Hadamard gate rotates a state on the Bloch sphere."
    ) == "hadamard_gate"


def test_specific_gate_diagrams_are_selected():
    cases = {
        "Explain Pauli-X gate": "pauli_x_gate",
        "Explain Pauli-Y gate": "pauli_y_gate",
        "Explain Pauli-Z gate": "pauli_z_gate",
        "Explain CNOT gate": "cnot_gate",
        "Explain SWAP gate": "swap_gate",
        "Explain Toffoli gate": "toffoli_gate",
    }
    for question, expected in cases.items():
        assert detect_diagram(question) == expected


def test_hadamard_diagram_renders():
    image = render_diagram("hadamard_gate")
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(image) > 5000


def test_algorithm_intent_beats_supporting_gate_terms():
    from app.quantum_visualizer import select_visualizations
    result = select_visualizations(
        "Explain Shor's algorithm with circuit diagram",
        "Hadamard gates create superposition and QFT extracts the period."
    )
    assert result["visualization_type"] == "shor_circuit"
    assert result["confidence"] > 0.9


def test_algorithm_visualizers_are_registered_and_renderable():
    from app.quantum_visualizer import VISUALIZATION_REGISTRY
    for kind in ("shor_circuit", "grover_circuit", "deutsch_jozsa_circuit",
                 "qft_circuit", "qpe_circuit", "vqe_circuit", "qaoa_circuit"):
        assert kind in SUPPORTED
        assert render_diagram(kind).startswith(b"\x89PNG\r\n\x1a\n")


def test_unsupported_named_concept_does_not_fallback_to_answer_keyword():
    assert detect_diagram(
        "Explain quantum error correction",
        "Hadamard gates and Bloch spheres are sometimes used in examples."
    ) is None


def test_supported_algorithm_regressions():
    cases = {
        "Explain Grover's algorithm": "grover_circuit",
        "Explain Deutsch-Jozsa algorithm": "deutsch_jozsa_circuit",
        "Explain QFT": "qft_circuit",
        "Explain quantum teleportation": "teleportation_circuit",
        "Explain Pauli-X gate": "pauli_x_gate",
    }
    for question, expected in cases.items():
        assert detect_diagram(question) == expected


def test_referential_followup_can_use_answer_context():
    from app.quantum_visualizer import select_visualizations
    assert select_visualizations("Explain it", "The Bloch sphere represents a qubit state.")["visualization_type"] == "bloch_sphere"


def test_shor_endpoint_returns_algorithm_specific_metadata():
    from app.routes.v3 import DiagramRequest, auto_visualization
    response = auto_visualization(DiagramRequest(
        question="Explain Shor's algorithm with circuit diagram",
        answer="Hadamard gates and QFT are important steps."
    ))
    assert response.headers["X-Diagram-Type"] == "shor_circuit"
    assert "Shor" in response.headers["X-Diagram-Title"]
    assert response.body.startswith(b"\x89PNG\r\n\x1a\n")


def test_phase_and_rotation_gates_are_registered():
    cases = {
        "Explain S gate": "s_gate", "Explain T gate": "t_gate",
        "Explain RX gate": "rx_gate", "Explain RY gate": "ry_gate",
        "Explain RZ gate": "rz_gate", "Explain CZ gate": "cz_gate",
    }
    for question, expected in cases.items():
        assert detect_diagram(question) == expected
        assert render_diagram(expected).startswith(b"\x89PNG\r\n\x1a\n")
