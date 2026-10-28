import math

import pytest

from slime_plugins.dcc.graphpo import GraphPOGraph, cosine, graph_advantages, normalize


def test_weighted_own_counts_are_not_pooled_recursively():
    graph = GraphPOGraph()
    root = graph.add_node(None, "question", [0, 0, 1])
    a = graph.add_node(root, "same progress", [1, 0, 0])
    b = graph.add_node(root, "same progress", [1, 0, 0])
    graph.add_node(a, "correct A", [0, 1, 0], terminal_success=1)
    graph.add_node(a, "incorrect A", [0, -1, 0], terminal_success=0)
    graph.add_node(b, "correct B", [0, 1, 0], terminal_success=1)
    assert graph.merge_pair(a, b)
    _, report = graph_advantages(graph)
    nodes = {node["id"]: node for node in report["nodes"]}
    assert nodes[root]["own_count"] == 3
    assert nodes[root]["own_correct"] == 2
    assert nodes[root]["score"] == pytest.approx(2 / 3)
    assert nodes[a]["score"] == pytest.approx(1.7 / 2.7)
    assert nodes[b]["score"] == pytest.approx(1.7 / 2.4)
    edges = {edge["edge"]: edge for edge in report["edges"]}
    assert edges[5]["step_reward"] == pytest.approx(1 - 1.7 / 2.4)
    group = next(item for item in report["correctness_groups"] if item["source_class"] == [a, b])
    assert group["edges"] == [3, 4, 5]
    assert len(set(group["edges"])) == 3


def test_no_pooling_matches_own_tree_values():
    graph = GraphPOGraph()
    root = graph.add_node(None, "root", [0, 1])
    good = graph.add_node(root, "good", [1, 0], terminal_success=1)
    bad = graph.add_node(root, "bad", [1, 0], terminal_success=0)
    graph.merge_pair(good, bad)
    _, report = graph_advantages(graph, pooling_weight=0)
    assert [node["score"] for node in report["nodes"]] == [0.5, 1, 0]


def test_same_embedding_at_edge_endpoints_gates_reward_to_zero():
    graph = GraphPOGraph()
    root = graph.add_node(None, "root", [1, 0])
    graph.add_node(root, "no semantic change", [1, 0], terminal_success=1)
    graph.add_node(root, "different", [0, 1], terminal_success=0)
    _, report = graph_advantages(graph)
    assert report["edges"][0]["novelty"] == 0
    assert report["edges"][0]["step_reward"] == 0


def test_ancestor_merge_is_rejected_even_at_identical_embedding():
    graph = GraphPOGraph()
    root = graph.add_node(None, "root", [1, 0])
    child = graph.add_node(root, "child", [1, 0])
    assert not graph.merge_pair(root, child)
    assert graph.merge_log[-1]["reason"] == "ancestor_descendant"


def test_pairwise_noncausal_merges_cannot_create_transitive_quotient_cycle():
    graph = GraphPOGraph()
    root = graph.add_node(None, "root", [0, 0, 1])
    a1 = graph.add_node(root, "A first path", [1, 0, 0])
    b1 = graph.add_node(a1, "B first path", [0, 1, 0])
    b2 = graph.add_node(root, "B second path", [0, 1, 0])
    a2 = graph.add_node(b2, "A second path", [1, 0, 0])
    assert graph.merge_pair(a1, a2)
    assert not graph.merge_pair(b1, b2)
    assert graph.merge_log[-1]["reason"] == "quotient_cycle"


def test_distinct_depth_efficiency_credit_favors_short_route():
    graph = GraphPOGraph()
    root = graph.add_node(None, "root", [0, 0, 1])
    short = graph.add_node(root, "same success", [1, 0, 0], terminal_success=1)
    detour = graph.add_node(root, "detour", [0, 1, 0])
    long = graph.add_node(detour, "same success", [1, 0, 0], terminal_success=1)
    assert graph.merge_pair(short, long)
    advantages, report = graph_advantages(graph)
    group = report["efficiency_groups"][0]
    assert group["common_ancestor"] == root
    assert group["lengths"] == [1, 2]
    assert group["routes"] == [[short], [detour, long]]
    assert advantages[short] == pytest.approx(0.5 / (0.5 + 1e-6))
    assert advantages[detour] == pytest.approx(-0.5 / (0.5 + 1e-6))
    assert advantages[long] == pytest.approx(advantages[detour])


def test_all_failure_evidence_does_not_invent_process_success():
    graph = GraphPOGraph()
    root = graph.add_node(None, "root", [0, 1])
    a = graph.add_node(root, "failure", [1, 0], terminal_success=0)
    detour = graph.add_node(root, "detour", [1, 1])
    b = graph.add_node(detour, "failure", [1, 0], terminal_success=0)
    graph.merge_pair(a, b)
    advantages, report = graph_advantages(graph)
    assert all(value == 0 for value in advantages.values())
    assert report["efficiency_groups"][0]["gate"] == 0


def test_absent_terminal_evidence_is_unknown_not_zero_reward():
    graph = GraphPOGraph()
    root = graph.add_node(None, "root", [0, 1])
    graph.add_node(root, "budget ended", [1, 0])
    advantages, report = graph_advantages(graph)
    assert advantages == {}
    assert all(node["score"] is None for node in report["nodes"])
    assert report["unestimated_edges"] == [1]


def test_early_stopped_node_receives_equivalent_partner_evidence():
    graph = GraphPOGraph()
    root = graph.add_node(None, "root", [0, 1])
    stopped = graph.add_node(root, "same state", [1, 0])
    expanded = graph.add_node(root, "same state", [1, 0])
    graph.add_node(expanded, "success", [0, -1], terminal_success=1)
    graph.merge_pair(stopped, expanded)
    _, report = graph_advantages(graph)
    assert report["nodes"][stopped]["own_count"] == 0
    assert report["nodes"][stopped]["score"] == 1


def test_partial_graph_keeps_supported_efficiency_on_unestimated_intermediate_edges():
    graph = GraphPOGraph()
    root = graph.add_node(None, "root", [0, 0, 0, 1])
    short = graph.add_node(root, "same state", [1, 0, 0, 0])
    graph.add_node(short, "success", [0, 1, 0, 0], terminal_success=1)
    detour1 = graph.add_node(root, "detour 1", [0, 0, 1, 0])
    detour2 = graph.add_node(detour1, "detour 2", [0, 1, 1, 0])
    long = graph.add_node(detour2, "same state", [1, 0, 0, 0])
    assert graph.merge_pair(short, long)
    advantages, report = graph_advantages(graph)
    rows = {row["edge"]: row for row in report["edges"]}
    assert advantages[short] == pytest.approx(1 / (1 + 1e-6))
    for edge in [detour1, detour2, long]:
        assert advantages[edge] == pytest.approx(-1 / (1 + 1e-6))
        assert rows[edge]["correctness_advantage"] is None
        assert rows[edge]["step_reward"] is None
        assert not rows[edge]["correctness_estimated"]
        assert rows[edge]["efficiency_supported"]
    assert report["unsupported_edges"] == []


@pytest.mark.parametrize("status", ["infrastructure_error", "model_protocol_failure", "context_budget_failure"])
def test_external_failures_cannot_be_shortcut_equivalence_targets(status):
    graph = GraphPOGraph()
    root = graph.add_node(None, "root", [0, 0, 1])
    broken = graph.add_node(root, "same apparent state", [1, 0, 0], closed=True, status=status)
    detour = graph.add_node(root, "detour", [0, 1, 0])
    good = graph.add_node(detour, "same apparent state", [1, 0, 0], terminal_success=1)
    assert not graph.merge_pair(broken, good)
    assert graph.merge_log[-1]["reason"] == "excluded_failure"
    advantages, report = graph_advantages(graph)
    assert broken not in advantages
    assert report["nodes"][broken]["score"] is None
    assert report["efficiency_groups"] == []
    assert all(edge["efficiency_advantage"] == 0 for edge in report["edges"])


def test_adaptive_budget_halves_redundant_and_reallocates_without_growth():
    graph = GraphPOGraph()
    root = graph.add_node(None, "root", [0, 0, 1])
    first = graph.add_node(root, "known", [1, 0, 0])
    duplicate = graph.add_node(root, "known again", [1, 0, 0])
    novel = graph.add_node(root, "novel", [0, 1, 0])
    graph.merge_pair(first, duplicate)
    budget, report = graph.allocate([first, duplicate, novel], 4)
    assert budget == {first: 5, duplicate: 2, novel: 5}
    assert sum(budget.values()) == 12
    assert report["saved"] == report["reallocated"] == 2
    budget, _ = graph.allocate([first, duplicate, novel], 1)
    assert budget == {first: 1, duplicate: 1, novel: 1}


def test_no_novel_frontier_does_not_claim_saved_budget_was_reallocated():
    graph = GraphPOGraph()
    root = graph.add_node(None, "root", [0, 1])
    earlier = graph.add_node(root, "known", [1, 0], terminal_success=1)
    duplicate = graph.add_node(root, "known", [1, 0])
    graph.merge_pair(earlier, duplicate)
    budget, report = graph.allocate([duplicate], 4)
    assert budget[duplicate] == 2
    assert report["saved"] == report["unused"] == 2
    assert report["reallocated"] == 0


def test_invalid_embedding_and_reward_fail_loudly():
    with pytest.raises(ValueError):
        cosine([0, 0], [1, 0])
    with pytest.raises(ValueError):
        cosine([math.nan], [1])
    with pytest.raises(ValueError):
        normalize([0, math.inf])
    graph = GraphPOGraph()
    graph.add_node(None, "root", [1, 0])
    with pytest.raises(ValueError):
        graph.add_node(0, "scaled reward", [0, 1], terminal_success=10)


def test_equation_independent_fixture_for_correctness_and_efficiency():
    graph = GraphPOGraph()
    root = graph.add_node(None, "root", [1, 0, 0, 0])
    a = graph.add_node(root, "A", [0, 1, 0, 0])
    b = graph.add_node(root, "B", [0, 0, 1, 0])
    graph.add_node(a, "right", [0, 0, 0, 1], terminal_success=1)
    graph.add_node(a, "wrong", [0, 0, 0, -1], terminal_success=0)
    graph.add_node(b, "wrong B", [0, 0, 0, -1], terminal_success=0)
    got, _ = graph_advantages(graph)
    # No merges: root=1/3, A=1/2, B=0; outgoing root rewards=1/6,-1/3.
    root_positive = 0.25 / (0.25 + 1e-6)
    assert got[a] == pytest.approx(root_positive)
    assert got[b] == pytest.approx(-root_positive)
    assert got[3] == pytest.approx(0.5 / (0.5 + 1e-6))
    assert got[4] == pytest.approx(-0.5 / (0.5 + 1e-6))
    assert got[5] == 0
