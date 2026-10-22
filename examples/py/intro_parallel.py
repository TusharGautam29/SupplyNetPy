"""Parallel simulation: the smallest working example.

A factory supplies two warehouses; each warehouse supplies two shops. The
links from the warehouses to the shops have a message delay of 1 time unit:
an order takes 1 unit to reach the warehouse, and the warehouse's answer and
shipping notice take 1 unit each to come back.

Those delays are what let the network be split: the factory and warehouses
run in one process, and each warehouse's shops in another. The script runs the
network both ways, in one process and in 3 processes, and prints the results
side by side. There is no randomness in this network, so they are identical.

Run:
  python intro_parallel.py
"""
import SupplyNetPy.Components as scm


# The network has to be described by a function at the top level of a file:
# each process calls it to build its own copy of the network.
def build():
    nodes = [{"ID": "F", "name": "Factory", "node_type": "infinite_supplier"}]
    links, demands = [], []
    for w in ("W1", "W2"):
        nodes.append({"ID": w, "name": w, "node_type": "warehouse", "capacity": 400, "initial_level": 200,
                      "inventory_holding_cost": 0.1, "replenishment_policy": scm.SSReplenishment,
                      "policy_param": {"s": 100, "S": 300}, "product_buy_price": 5, "product_sell_price": 8})
        links.append({"ID": f"F-{w}", "source": "F", "sink": w, "cost": 10, "lead_time": lambda: 3})
        for s in ("a", "b"):
            shop = f"{w}{s}"
            nodes.append({"ID": shop, "name": shop, "node_type": "shop", "capacity": 80, "initial_level": 40,
                          "inventory_holding_cost": 0.2, "replenishment_policy": scm.SSReplenishment,
                          "policy_param": {"s": 30, "S": 80}, "product_buy_price": 8, "product_sell_price": 12})
            links.append({"ID": f"{w}-{shop}", "source": w, "sink": shop, "cost": 2,
                          "lead_time": lambda: 2, "message_delay": 1})
            demands.append({"ID": f"C-{shop}", "name": "customers", "demand_node": shop,
                            "order_arrival_model": lambda gap=(0.9 if s == "a" else 1.4): gap,
                            "order_quantity_model": lambda: 5})
    return nodes, links, demands


# Needed when starting processes: each new process re-reads this file, and must
# not run the simulation again itself.
if __name__ == "__main__":
    one = scm.simulate_sc_net(scm.create_sc_net(*build()), sim_time=200, logging=False)

    pieces = {"F": 0, "W1": 0, "W2": 0, "W1a": 1, "W1b": 1, "W2a": 2, "W2b": 2}
    par = scm.simulate_sc_net_parallel(build, sim_time=200, assignment=pieces, logging=False)

    print(f"split: {par['parallel']['assignment']}")
    print(f"links between pieces: {par['parallel']['cut_links']}")
    print(f"window length: {par['parallel']['lookahead']}, windows run: {par['parallel']['windows']}\n")
    print(f"{'':20}{'one process':>18}{'3 processes':>18}")
    for key in ("total_demand", "shortage", "backorders", "revenue", "total_cost", "profit"):
        a, b = one[key], par[key]
        if isinstance(a, float):
            a, b = round(a, 2), round(b, 2)
        print(f"{key:20}{str(a):>18}{str(b):>18}")
    print(f"\nW1 statistics match: {one['nodes']['W1'].stats.get_statistics() == par['nodes']['W1'].stats.get_statistics()}")
