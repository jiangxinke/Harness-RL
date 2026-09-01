from argparse import Namespace

import pytest


class _RemotePortProbe:
    def remote(self, start_port=10000, consecutive=1):
        return "127.0.0.1", start_port


class _Engine:
    _get_current_node_ip_and_free_port = _RemotePortProbe()


def _allocate(monkeypatch, base_port):
    from slime.ray import rollout

    monkeypatch.setattr(rollout.ray, "get", lambda value: value)
    args = Namespace(
        rollout_num_gpus_per_engine=1,
        num_gpus_per_node=4,
        sglang_dp_size=1,
    )
    return rollout._allocate_rollout_engine_addr_and_ports_normal(
        args=args,
        rollout_engines=[(rank, _Engine()) for rank in range(4)],
        base_port=base_port,
        port_limit=base_port + 1024,
    )[0]


def test_independent_rollout_managers_use_disjoint_configured_port_blocks(monkeypatch):
    first = _allocate(monkeypatch, 15000)
    second = _allocate(monkeypatch, 22000)

    def allocated_port_numbers(mapping):
        allocated = set()
        for ports in mapping.values():
            allocated.add(ports["port"])
            allocated.add(ports["nccl_port"])
            allocated.add(int(ports["dist_init_addr"].rsplit(":", 1)[1]))
        return allocated

    first_ports = allocated_port_numbers(first)
    second_ports = allocated_port_numbers(second)
    assert not (first_ports & second_ports)
    assert first[0]["port"] == 15000
    assert second[0]["port"] == 22000


def test_engine_port_allocator_fails_closed_outside_reserved_block(monkeypatch):
    from slime.ray import rollout

    monkeypatch.setattr(rollout.ray, "get", lambda value: value)
    args = Namespace(
        rollout_num_gpus_per_engine=1,
        num_gpus_per_node=1,
        sglang_dp_size=1,
    )
    with pytest.raises(RuntimeError, match="exhausted the reserved block"):
        rollout._allocate_rollout_engine_addr_and_ports_normal(
            args=args,
            rollout_engines=[(0, _Engine())],
            base_port=15000,
            port_limit=15002,
        )
