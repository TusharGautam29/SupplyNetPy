"""
Splitting one supply-chain network into pieces for parallel simulation.

The idea: cut the network into a few pieces ("partitions") and run each piece
on its own CPU core, with its own SimPy environment. The pieces talk only by
sending messages along the links that connect them.

A link may only be cut if it has a ``message_delay`` above 0. That delay is
what lets the pieces run side by side safely: a message sent at time t cannot
arrive before t + delay, so each piece can run ahead by that much without
waiting for the others. A link with no delay keeps its two nodes in the same
piece.

This module provides:

- ``partition_network``: decide which piece each node goes into.
- ``check_partition``: check a split (yours or ours) and report on it.
- ``build_partition``: build the part of the network that belongs to one piece.
  Nodes that live in another piece are represented by a ``RemoteNode``
  stand-in.

Running the pieces together (sending the messages between processes) is the
job of the parallel runner, which is not written yet. See ``PDES_DESIGN.md``.
"""
import simpy

from SupplyNetPy.Components.core import Node, Link, Demand
from SupplyNetPy.Components.utilities import _build_node

__all__ = ["partition_network", "check_partition", "build_partition"]


def _require_dicts(nodes, links, demands):
    # Each piece is rebuilt from the descriptions inside its own process, so
    # parallel runs need the dict style; ready-made objects can't be split up.
    for name, items in (("nodes", nodes), ("links", links), ("demands", demands)):
        if not all(isinstance(x, dict) for x in items):
            raise ValueError(f"Parallel simulation needs {name} given as dicts (the netlist style), not ready-made objects.")


def _delay(link):
    return link.get("message_delay", 0) or 0


def _node_weights(nodes, links, demands):
    """
    Rough guess of how much simulation work each node creates.

    A node gets 1, plus 1 for each link it touches (orders come and go along
    links), plus 2 for each customer demand attached to it (customer arrivals
    are usually the busiest stream of events).
    """
    # shortcut: weights are a structural guess, not measured event counts; upgrade
    # to counts from a short serial pilot run if pieces come out unbalanced.
    weight = {n["ID"]: 1 for n in nodes}
    for link in links:
        weight[link["source"]] += 1
        weight[link["sink"]] += 1
    for d in demands:
        weight[d["demand_node"]] += 2
    return weight


def partition_network(nodes, links, demands, n_parts, balance_tolerance=0.1):
    """
    Decide which piece (partition) each node goes into.

    The goals are: (1) give every piece about the same amount of work, so no
    core sits idle waiting for a slower one; (2) cut as few links as possible,
    because every cut link means messages between cores.

    How it works:

    1. Nodes joined by a link with no message delay are glued together into
       one group. A group always stays in one piece.
    2. Pieces are grown one at a time: a piece keeps taking the free group
       best tied to it (most links into the piece, fewest links elsewhere)
       until it holds about its fair share of the
       work.
    3. Then groups are moved, one at a time, to a neighbouring piece whenever
       that cuts fewer links and keeps the work balanced. This repeats until
       no move helps.

    The result is the same every time for the same input.

    Parameters:
        nodes (list): Node descriptions (dicts), as for ``create_sc_net``.
        links (list): Link descriptions (dicts). ``message_delay`` (default 0)
            decides which links may be cut.
        demands (list): Demand descriptions (dicts). A demand always goes in the
            same piece as the node it buys from.
        n_parts (int): Number of pieces to make (usually the number of cores).
        balance_tolerance (float): How much more work than the average a piece
            may get while moving groups to cut fewer links (0.1 = 10% more).

    Returns:
        dict: ``{node_ID: piece_number}``, with piece numbers ``0 .. n_parts-1``.

    Raises:
        ValueError: If the descriptions are not dicts, or if there are fewer
            groups than ``n_parts`` (add a ``message_delay`` to more links).
    """
    _require_dicts(nodes, links, demands)
    if not isinstance(n_parts, int) or n_parts < 1:
        raise ValueError("n_parts must be a whole number of at least 1.")

    # Step 1: glue nodes joined by no-delay links into groups (union-find:
    # every node points to a "leader"; nodes with the same leader are a group).
    leader = {n["ID"]: n["ID"] for n in nodes}

    def find(x):
        while leader[x] != x:
            leader[x] = leader[leader[x]]
            x = leader[x]
        return x

    for link in links:
        if _delay(link) == 0:
            a, b = find(link["source"]), find(link["sink"])
            if a != b:
                leader[max(a, b)] = min(a, b)  # smaller ID leads, so results don't depend on link order

    node_weight = _node_weights(nodes, links, demands)
    group_of = {nid: find(nid) for nid in leader}
    groups = sorted(set(group_of.values()))
    if len(groups) < n_parts:
        raise ValueError(
            f"Only {len(groups)} group(s) of nodes can be separated, but {n_parts} pieces were asked for. "
            f"Give more links a message_delay above 0, or use fewer pieces.")

    group_weight = {g: 0 for g in groups}
    for nid, g in group_of.items():
        group_weight[g] += node_weight[nid]
    # How many links join each pair of groups (only delayed links can join two groups).
    neighbours = {g: {} for g in groups}
    for link in links:
        a, b = group_of[link["source"]], group_of[link["sink"]]
        if a != b:
            neighbours[a][b] = neighbours[a].get(b, 0) + 1
            neighbours[b][a] = neighbours[b].get(a, 0) + 1

    # Step 2: grow the pieces one at a time, like a spreading region. A piece
    # starts from the heaviest group still free and keeps taking the free
    # group that is best tied to it -- most links into the piece, fewest links
    # going elsewhere -- until it holds about its fair share of the work.
    # This keeps neighbours together, so few links are cut. The last piece
    # takes whatever is left.
    piece = {}
    load = [0] * n_parts
    target = sum(group_weight.values()) / n_parts
    degree = {g: sum(neighbours[g].values()) for g in groups}
    free = sorted(groups)
    for p in range(n_parts):
        if p == n_parts - 1:
            for g in free:
                piece[g] = p
                load[p] += group_weight[g]
            break
        links_in = {g: 0 for g in free}  # links from each free group into piece p
        while len(free) > n_parts - 1 - p:  # leave at least one group for each later piece
            # links into the piece minus links elsewhere; ties: heavier, then first in sorted order
            g = max(free, key=lambda g: (2 * links_in[g] - degree[g], group_weight[g]))
            w = group_weight[g]
            # Stop if adding g would land further from the fair share than stopping now.
            if load[p] > 0 and abs(load[p] + w - target) >= abs(load[p] - target):
                break
            piece[g] = p
            load[p] += w
            free.remove(g)
            del links_in[g]
            for h, count in neighbours[g].items():
                if h in links_in:
                    links_in[h] += count
            if load[p] >= target:
                break

    # Step 3: move a group to another piece when that cuts fewer links,
    # without overloading the piece it moves to or emptying the one it leaves.
    # Every move cuts strictly fewer links, so this always finishes.
    cap = (1 + balance_tolerance) * sum(load) / n_parts
    for _ in range(50):
        moved = False
        for g in groups:
            p, w = piece[g], group_weight[g]
            if load[p] - w <= 0:
                continue
            links_to = [0] * n_parts
            for h, count in neighbours[g].items():
                links_to[piece[h]] += count
            best = None
            for q in range(n_parts):
                if q != p and load[q] + w <= cap and links_to[q] > links_to[p]:
                    if best is None or links_to[q] > links_to[best]:
                        best = q
            if best is not None:
                piece[g] = best
                load[p] -= w
                load[best] += w
                moved = True
        if not moved:
            break

    return {nid: piece[g] for nid, g in group_of.items()}


def check_partition(nodes, links, demands, assignment):
    """
    Check a split of the network into pieces and report on it.

    Use it on the output of ``partition_network`` or on a split you made by
    hand (for example, one piece per echelon).

    Parameters:
        nodes (list): Node descriptions (dicts).
        links (list): Link descriptions (dicts).
        demands (list): Demand descriptions (dicts).
        assignment (dict): ``{node_ID: piece_number}`` for every node.

    Returns:
        dict: A report with
            ``n_parts`` (number of pieces),
            ``loads`` (estimated work per piece, same guess as ``partition_network``),
            ``cut_links`` (IDs of links that join two different pieces), and
            ``lookahead`` (the smallest ``message_delay`` among the cut links:
            how far ahead each piece may safely run; ``inf`` if nothing is cut).

    Raises:
        ValueError: If a node has no piece, piece numbers are not
            ``0 .. n-1``, or a link with no message delay joins two pieces.
    """
    _require_dicts(nodes, links, demands)
    missing = [n["ID"] for n in nodes if n["ID"] not in assignment]
    if missing:
        raise ValueError(f"These nodes are not assigned to a piece: {missing}")
    n_parts = max(assignment.values()) + 1
    if sorted(set(assignment.values())) != list(range(n_parts)):
        raise ValueError("Piece numbers must be 0, 1, 2, ... with no gaps.")

    bad = [l["ID"] for l in links if assignment[l["source"]] != assignment[l["sink"]] and _delay(l) == 0]
    if bad:
        raise ValueError(
            f"These links join two different pieces but have no message_delay: {bad}. "
            f"A link between pieces needs a message_delay above 0.")

    weight = _node_weights(nodes, links, demands)
    loads = [0] * n_parts
    for nid, w in weight.items():
        loads[assignment[nid]] += w
    cut = [l for l in links if assignment[l["source"]] != assignment[l["sink"]]]
    return {
        "n_parts": n_parts,
        "loads": loads,
        "cut_links": [l["ID"] for l in cut],
        "lookahead": min((_delay(l) for l in cut), default=float("inf")),
    }


class RemoteNode(Node):
    """
    Stand-in for a node that lives in another piece of a partitioned network.

    When a link is cut, the buyer's piece still needs an object at the
    supplier end of the link. This stand-in plays that role. It knows only the
    supplier's fixed facts that the buyer reads (ID, name, type, selling price,
    raw material) plus a copy of its stock level, which the parallel runner
    will refresh. It has no real stock and no statistics. Orders to it travel
    as messages, so asking it to handle an order directly is an error.

    Parameters:
        env (simpy.Environment): The environment of the piece holding the stand-in.
        node (dict): The description of the real node.
        home_part (int): The piece where the real node lives.

    Attributes:
        home_part (int): The piece where the real node lives.
        raw_material (RawMaterial or None): The raw material the real node supplies, if any.
        sell_price (float): The real node's selling price.
        inventory: Object whose ``level`` is the last known stock level of the real node.
    """

    class _StockLevel:
        def __init__(self, level):
            self.level = level

    def __init__(self, env, node, home_part):
        super().__init__(env=env, ID=node["ID"], name=node["name"], node_type=node["node_type"],
                         logging=False, logger_name=f"remote.{node['ID']}")
        self.home_part = home_part
        self.raw_material = node.get("raw_material")
        # Same selling price rules as the real classes: a Supplier sells at its
        # raw material's cost; every other node has a product_sell_price.
        if "supplier" in self.node_type:
            self.sell_price = self.raw_material.cost if self.raw_material else 0
        else:
            self.sell_price = node.get("product_sell_price", 0)
        start = float("inf") if self.node_type == "infinite_supplier" else node.get("initial_level", 0)
        self.inventory = RemoteNode._StockLevel(start)

    def receive_order(self, quantity, signal_drop=True):
        raise RuntimeError(
            f"{self.ID} lives in piece {self.home_part}; orders to it must travel as messages "
            f"between pieces. Run partitioned networks with the parallel runner.")


def build_partition(nodes, links, demands, assignment, part, env=None):
    """
    Build the part of the network that belongs to one piece.

    The piece gets: its own nodes (real objects), a ``RemoteNode`` stand-in for
    every node in another piece that supplies one of its nodes, every link
    whose receiving end (sink) is in this piece, and every demand on its nodes.

    Parameters:
        nodes (list): Node descriptions (dicts) for the whole network.
        links (list): Link descriptions (dicts) for the whole network.
        demands (list): Demand descriptions (dicts) for the whole network.
        assignment (dict): ``{node_ID: piece_number}``, e.g. from ``partition_network``.
        part (int): Which piece to build.
        env (simpy.Environment, optional): Environment for this piece. A new one is made if not given.

    Returns:
        dict: ``nodes``, ``links``, ``demands`` (this piece's own objects, keyed by ID),
        ``remote_nodes`` (stand-ins, keyed by ID),
        ``outbound_links`` (descriptions of cut links whose supplier is in this
        piece: the runner uses them to send answers back), ``env`` and ``part``.

    Raises:
        ValueError: If ``check_partition`` rejects the assignment.
    """
    check_partition(nodes, links, demands, assignment)
    env = env or simpy.Environment()
    by_id = {n["ID"]: n for n in nodes}

    local = {n["ID"]: _build_node(env, n) for n in nodes if assignment[n["ID"]] == part}
    remote = {}
    built_links = {}
    outbound = []
    for link in links:
        src, snk = link["source"], link["sink"]
        if assignment[snk] != part:
            if assignment[src] == part:
                outbound.append(link)
            continue
        if src not in local and src not in remote:
            remote[src] = RemoteNode(env, by_id[src], assignment[src])
        source = local[src] if src in local else remote[src]
        params = {k: v for k, v in link.items() if k not in ("source", "sink")}
        built_links[link["ID"]] = Link(env=env, source=source, sink=local[snk], **params)

    built_demands = {}
    for d in demands:
        if assignment[d["demand_node"]] == part:
            params = {k: v for k, v in d.items() if k not in ("demand_node", "node_type")}
            built_demands[d["ID"]] = Demand(env=env, demand_node=local[d["demand_node"]], **params)

    return {"nodes": local, "links": built_links, "demands": built_demands,
            "remote_nodes": remote, "outbound_links": outbound, "env": env, "part": part}
