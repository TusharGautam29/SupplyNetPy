import pytest
import simpy

import SupplyNetPy.Components as scm
from SupplyNetPy.Components.parallel import RemoteNode, _with_delays


def _small_net():
    """S0 -> D0, D1 (delay 1). D0 -> R0, R1 and D1 -> R2, R3 (no delay).
    One cross link D1 -> R1 (delay 2). So the groups that must stay together
    are {S0}, {D0, R0, R1}, {D1, R2, R3}."""
    rm = scm.RawMaterial(ID="RM", name="raw", extraction_quantity=100, extraction_time=1, mining_cost=1, cost=4)
    nodes = [{"ID": "S0", "name": "S0", "node_type": "infinite_supplier", "raw_material": rm}]
    for d in ("D0", "D1"):
        nodes.append({"ID": d, "name": d, "node_type": "distributor", "capacity": 300, "initial_level": 150,
                      "inventory_holding_cost": 0.1, "replenishment_policy": scm.SSReplenishment,
                      "policy_param": {"s": 100, "S": 300}, "product_buy_price": 5, "product_sell_price": 7})
    for r in ("R0", "R1", "R2", "R3"):
        nodes.append({"ID": r, "name": r, "node_type": "retailer", "capacity": 100, "initial_level": 50,
                      "inventory_holding_cost": 0.1, "replenishment_policy": scm.SSReplenishment,
                      "policy_param": {"s": 30, "S": 100}, "product_buy_price": 7, "product_sell_price": 9})
    links = [
        {"ID": "L_S0_D0", "source": "S0", "sink": "D0", "cost": 1, "lead_time": lambda: 2, "message_delay": 1},
        {"ID": "L_S0_D1", "source": "S0", "sink": "D1", "cost": 1, "lead_time": lambda: 2, "message_delay": 1},
        {"ID": "L_D0_R0", "source": "D0", "sink": "R0", "cost": 1, "lead_time": lambda: 1},
        {"ID": "L_D0_R1", "source": "D0", "sink": "R1", "cost": 1, "lead_time": lambda: 1},
        {"ID": "L_D1_R2", "source": "D1", "sink": "R2", "cost": 1, "lead_time": lambda: 1},
        {"ID": "L_D1_R3", "source": "D1", "sink": "R3", "cost": 1, "lead_time": lambda: 1},
        {"ID": "L_D1_R1", "source": "D1", "sink": "R1", "cost": 1, "lead_time": lambda: 1, "message_delay": 2},
    ]
    demands = [{"ID": f"d_{r}", "name": f"d_{r}", "order_arrival_model": lambda: 1,
                "order_quantity_model": lambda: 5, "demand_node": r} for r in ("R0", "R1", "R2", "R3")]
    return nodes, links, demands


def _two_clusters():
    """Two identical clusters, every link delayed, joined by a single link
    DA -> RB0. The best 2-way split cuts only that one link."""
    nodes, links, demands = [], [], []
    for c in ("A", "B"):
        nodes.append({"ID": f"S{c}", "name": f"S{c}", "node_type": "infinite_supplier"})
        nodes.append({"ID": f"D{c}", "name": f"D{c}", "node_type": "distributor", "capacity": 100,
                      "initial_level": 50, "inventory_holding_cost": 0.1})
        links.append({"ID": f"L_S{c}_D{c}", "source": f"S{c}", "sink": f"D{c}", "cost": 1,
                      "lead_time": lambda: 1, "message_delay": 1})
        for i in range(3):
            r = f"R{c}{i}"
            nodes.append({"ID": r, "name": r, "node_type": "retailer", "capacity": 50,
                          "initial_level": 20, "inventory_holding_cost": 0.1})
            links.append({"ID": f"L_D{c}_{r}", "source": f"D{c}", "sink": r, "cost": 1,
                          "lead_time": lambda: 1, "message_delay": 1})
            demands.append({"ID": f"d_{r}", "name": "d", "order_arrival_model": lambda: 1,
                            "order_quantity_model": lambda: 1, "demand_node": r})
    links.append({"ID": "L_DA_RB0", "source": "DA", "sink": "RB0", "cost": 1,
                  "lead_time": lambda: 1, "message_delay": 1})
    return nodes, links, demands


class TestPartitionNetwork:
    def test_no_delay_links_keep_nodes_together(self):
        nodes, links, demands = _small_net()
        a = scm.partition_network(nodes, links, demands, 2)
        assert a["D0"] == a["R0"] == a["R1"]
        assert a["D1"] == a["R2"] == a["R3"]
        assert a["D0"] != a["D1"]
        assert set(a.values()) == {0, 1}

    def test_same_result_every_time(self):
        nodes, links, demands = _small_net()
        assert scm.partition_network(nodes, links, demands, 2) == scm.partition_network(nodes, links, demands, 2)

    def test_too_many_pieces_rejected(self):
        nodes, links, demands = _small_net()
        with pytest.raises(ValueError, match="group"):
            scm.partition_network(nodes, links, demands, 4)  # only 3 separable groups

    def test_objects_rejected(self):
        env = simpy.Environment()
        node = scm.Supplier(env=env, ID="S", name="S", node_type="infinite_supplier")
        with pytest.raises(ValueError, match="dicts"):
            scm.partition_network([node], [], [], 1)

    def test_cuts_the_single_bridge_between_two_clusters(self):
        nodes, links, demands = _two_clusters()
        a = scm.partition_network(nodes, links, demands, 2)
        report = scm.check_partition(nodes, links, demands, a)
        assert report["cut_links"] == ["L_DA_RB0"]

    def test_balances_work(self):
        nodes, links, demands = _two_clusters()
        a = scm.partition_network(nodes, links, demands, 2)
        loads = scm.check_partition(nodes, links, demands, a)["loads"]
        assert max(loads) <= 1.1 * sum(loads) / 2


class TestCheckPartition:
    def test_report(self):
        nodes, links, demands = _small_net()
        a = {"S0": 0, "D0": 0, "R0": 0, "R1": 0, "D1": 1, "R2": 1, "R3": 1}
        r = scm.check_partition(nodes, links, demands, a)
        assert r["n_parts"] == 2
        assert sorted(r["cut_links"]) == ["L_D1_R1", "L_S0_D1"]
        assert r["lookahead"] == 1  # smallest delay among the cut links
        assert sum(r["loads"]) == 7 + 2 * len(links) + 2 * len(demands)

    def test_single_piece_has_no_cut(self):
        nodes, links, demands = _small_net()
        r = scm.check_partition(nodes, links, demands, {n["ID"]: 0 for n in nodes})
        assert r["cut_links"] == [] and r["lookahead"] == float("inf")

    def test_cutting_a_no_delay_link_rejected(self):
        nodes, links, demands = _small_net()
        a = {"S0": 0, "D0": 0, "R0": 1, "R1": 0, "D1": 1, "R2": 1, "R3": 1}  # cuts L_D0_R0 (no delay)
        with pytest.raises(ValueError, match="L_D0_R0"):
            scm.check_partition(nodes, links, demands, a)

    def test_missing_node_and_gaps_rejected(self):
        nodes, links, demands = _small_net()
        a = {n["ID"]: 0 for n in nodes}
        del a["R3"]
        with pytest.raises(ValueError, match="R3"):
            scm.check_partition(nodes, links, demands, a)
        a = {n["ID"]: 0 for n in nodes}
        a["S0"] = 2  # piece 1 is missing
        with pytest.raises(ValueError, match="gaps"):
            scm.check_partition(nodes, links, demands, a)


class TestBuildPartition:
    ASSIGN = {"S0": 0, "D0": 0, "R0": 0, "R1": 0, "D1": 1, "R2": 1, "R3": 1}

    def test_every_node_link_and_demand_lands_in_exactly_one_piece(self):
        nodes, links, demands = _small_net()
        pieces = [scm.build_partition(nodes, links, demands, self.ASSIGN, p) for p in (0, 1)]
        node_ids = [nid for pc in pieces for nid in pc["nodes"]]
        link_ids = [lid for pc in pieces for lid in pc["links"]]
        demand_ids = [did for pc in pieces for did in pc["demands"]]
        assert sorted(node_ids) == sorted(n["ID"] for n in nodes)
        assert sorted(link_ids) == sorted(l["ID"] for l in links)
        assert sorted(demand_ids) == sorted(d["ID"] for d in demands)

    def test_stand_ins_for_remote_suppliers(self):
        nodes, links, demands = _small_net()
        p0 = scm.build_partition(nodes, links, demands, self.ASSIGN, 0)
        p1 = scm.build_partition(nodes, links, demands, self.ASSIGN, 1)
        # Piece 0's R1 buys from D1 (piece 1); piece 1's D1 buys from S0 (piece 0).
        assert set(p0["remote_nodes"]) == {"D1"}
        assert set(p1["remote_nodes"]) == {"S0"}
        assert p0["links"]["L_D1_R1"].source is p0["remote_nodes"]["D1"]
        assert [l["ID"] for l in p0["outbound_links"]] == ["L_S0_D1"]
        assert [l["ID"] for l in p1["outbound_links"]] == ["L_D1_R1"]

        d1 = p0["remote_nodes"]["D1"]
        assert isinstance(d1, RemoteNode) and d1.home_part == 1
        assert d1.node_type == "distributor" and d1.sell_price == 7 and d1.inventory.level == 150
        s0 = p1["remote_nodes"]["S0"]
        assert s0.sell_price == 4 and s0.raw_material.ID == "RM" and s0.inventory.level == float("inf")
        # Stand-ins are not counted as the piece's own nodes.
        assert "D1" not in p0["nodes"] and "S0" not in p1["nodes"]

    def test_stand_in_refuses_direct_orders(self):
        nodes, links, demands = _small_net()
        p0 = scm.build_partition(nodes, links, demands, self.ASSIGN, 0)
        with pytest.raises(RuntimeError, match="messages"):
            p0["remote_nodes"]["D1"].receive_order(10)

    def test_one_piece_matches_create_sc_net(self):
        """With every node in one piece, build_partition must build exactly
        the network create_sc_net builds: same KPIs at every node."""
        nodes, links, demands = _small_net()
        serial = scm.create_sc_net(nodes, links, demands)
        serial["env"].run(until=60)
        whole = scm.build_partition(nodes, links, demands, {n["ID"]: 0 for n in nodes}, 0)
        assert whole["remote_nodes"] == {} and whole["outbound_links"] == []
        whole["env"].run(until=60)
        for nid, node in serial["nodes"].items():
            assert whole["nodes"][nid].stats.get_statistics() == node.stats.get_statistics(), nid


def _chain_net():
    """Supplier -> manufacturer -> distributor -> two shops, every link delayed.
    Small stocks so orders often wait for stock (shortages and backorders)."""
    rm = scm.RawMaterial(ID="RM", name="raw", extraction_quantity=40, extraction_time=1, mining_cost=1, cost=2)
    prod = scm.Product(ID="P", name="prod", manufacturing_cost=3, manufacturing_time=1,
                       raw_materials=[(rm, 2)], sell_price=12, buy_price=6, batch_size=40)
    nodes = [
        {"ID": "S", "name": "S", "node_type": "supplier", "raw_material": rm, "capacity": 600, "initial_level": 300},
        {"ID": "M", "name": "M", "node_type": "manufacturer", "capacity": 200, "initial_level": 60,
         "inventory_holding_cost": 0.2, "replenishment_policy": scm.SSReplenishment,
         "policy_param": {"s": 80, "S": 200}, "product_sell_price": 12, "product": prod},
        {"ID": "D", "name": "D", "node_type": "distributor", "capacity": 120, "initial_level": 60,
         "inventory_holding_cost": 0.1, "replenishment_policy": scm.SSReplenishment,
         "policy_param": {"s": 50, "S": 120}, "product_buy_price": 12, "product_sell_price": 15},
    ]
    for r in ("R0", "R1"):
        nodes.append({"ID": r, "name": r, "node_type": "retailer", "capacity": 40, "initial_level": 20,
                      "inventory_holding_cost": 0.3, "replenishment_policy": scm.SSReplenishment,
                      "policy_param": {"s": 15, "S": 40}, "product_buy_price": 15, "product_sell_price": 20})
    links = [
        {"ID": "L_S_M", "source": "S", "sink": "M", "cost": 1, "lead_time": lambda: 1.5, "message_delay": 1},
        {"ID": "L_M_D", "source": "M", "sink": "D", "cost": 2, "lead_time": lambda: 2, "message_delay": 1.5},
        {"ID": "L_D_R0", "source": "D", "sink": "R0", "cost": 1, "lead_time": lambda: 1, "message_delay": 1},
        {"ID": "L_D_R1", "source": "D", "sink": "R1", "cost": 1, "lead_time": lambda: 0.5, "message_delay": 1.25},
    ]
    demands = [
        {"ID": "d_R0", "name": "d_R0", "order_arrival_model": lambda: 0.7, "order_quantity_model": lambda: 4, "demand_node": "R0"},
        {"ID": "d_R1", "name": "d_R1", "order_arrival_model": lambda: 1.3, "order_quantity_model": lambda: 6, "demand_node": "R1"},
    ]
    return nodes, links, demands


def _chain_no_delay():
    nodes, links, demands = _chain_net()
    return nodes, [{k: v for k, v in l.items() if k != "message_delay"} for l in links], demands


def _serial_stats(factory, sim_time):
    net = scm.create_sc_net(*factory())
    net["env"].run(until=sim_time)
    return {nid: node.stats.get_statistics() for nid, node in net["nodes"].items()}


class TestRunParallel:
    """A parallel run of a network without randomness must give exactly the
    same statistics as the one-process run with the same message delays."""

    CHAIN_SPLIT = {"S": 0, "M": 1, "D": 2, "R0": 0, "R1": 1}  # every link is cut

    def test_every_link_cut_matches_serial(self):
        serial = _serial_stats(_chain_net, 80)
        result = scm.run_parallel(_chain_net, 80, assignment=self.CHAIN_SPLIT, processes=False)
        assert result["lookahead"] == 1 and result["windows"] == 80
        assert result["stats"] == serial
        assert serial["D"]["shortage"][0] > 0, "the test network should make orders wait for stock"

    def test_mixed_delays_with_integer_times_matches_serial(self):
        serial = _serial_stats(_small_net, 60)
        result = scm.run_parallel(_small_net, 60, assignment=TestBuildPartition.ASSIGN, processes=False)
        assert result["stats"] == serial

    def test_separate_processes_match_serial(self):
        serial = _serial_stats(_chain_net, 80)
        result = scm.run_parallel(_chain_net, 80, assignment=self.CHAIN_SPLIT)
        assert result["stats"] == serial

    def test_partitioner_split_and_one_piece(self):
        serial = _serial_stats(_chain_net, 50)
        assert scm.run_parallel(_chain_net, 50, n_parts=2, processes=False)["stats"] == serial
        one = scm.run_parallel(_chain_net, 50, n_parts=1, processes=False)
        assert one["windows"] == 1 and one["stats"] == serial

    def test_needs_n_parts_or_assignment(self):
        with pytest.raises(ValueError, match="n_parts"):
            scm.run_parallel(_chain_net, 10)

    def test_network_totals_match_simulate_sc_net(self):
        serial = scm.simulate_sc_net(scm.create_sc_net(*_chain_net()), 80, logging=False)
        par = scm.simulate_sc_net_parallel(_chain_net, 80, assignment=self.CHAIN_SPLIT,
                                           processes=False, logging=False)
        kpis = [k for k in serial if k not in ("nodes", "links", "demands", "env")]
        assert {k: par[k] for k in kpis} == {k: serial[k] for k in kpis}
        assert serial["profit"] != 0
        for nid, node in serial["nodes"].items():
            assert par["nodes"][nid].stats.get_statistics() == node.stats.get_statistics(), nid
        for did, demand in serial["demands"].items():
            assert par["demands"][did].stats.get_statistics() == demand.stats.get_statistics(), did
        assert par["parallel"]["cut_links"] == ["L_S_M", "L_M_D", "L_D_R0", "L_D_R1"]

    def test_cut_delay_goes_only_on_cut_links(self):
        par = scm.simulate_sc_net_parallel(_chain_no_delay, 60, n_parts=2, cut_delay=1,
                                           processes=False, logging=False)
        cut = par["parallel"]["cut_links"]
        assert cut and par["parallel"]["lookahead"] == 1
        for lid, link in par["links"].items():
            assert link.message_delay == (1 if lid in cut else 0), lid
        serial = scm.simulate_sc_net(
            scm.create_sc_net(*_with_delays(_chain_no_delay, {lid: 1 for lid in cut})), 60, logging=False)
        assert par["profit"] == serial["profit"] and par["shortage"] == serial["shortage"]

    def test_cut_delay_must_be_positive(self):
        with pytest.raises(ValueError, match="cut_delay"):
            scm.simulate_sc_net_parallel(_chain_no_delay, 10, n_parts=2, cut_delay=0)

    def test_cut_link_outside_the_runner_refuses_orders(self):
        nodes, links, demands = _small_net()
        p0 = scm.build_partition(nodes, links, demands, TestBuildPartition.ASSIGN, 0)
        with pytest.raises(RuntimeError, match="run_parallel"):
            p0["links"]["L_D1_R1"].send_delivered(1)
