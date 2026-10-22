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
- ``run_parallel``: run all the pieces side by side, one process each, and
  carry the messages between them. See ``PDES_DESIGN.md``.
"""
import functools
import itertools
import multiprocessing
import traceback

import simpy

from SupplyNetPy.Components.core import Node, Link, Demand, NodeType, global_logger, _deliver_at
from SupplyNetPy.Components.utilities import (_build_node, create_sc_net, _node_summary,
                                              _fill_network_kpis, _log_network_kpis)

__all__ = ["partition_network", "check_partition", "build_partition", "run_parallel",
           "simulate_sc_net_parallel"]


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
    # shortcut: weights are a structural guess. Weighting by event counts from a
    # pilot run was tried and was slower (more cut links); see PDES_DESIGN.md phase 6.
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
    raw material) plus a copy of its stock level, which ``run_parallel``
    refreshes after each window. It has no real stock and no statistics. Orders to it travel
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
        cls = Link if src in local else _CutLink
        built_links[link["ID"]] = cls(env=env, source=source, sink=local[snk], **params)

    built_demands = {}
    for d in demands:
        if assignment[d["demand_node"]] == part:
            params = {k: v for k, v in d.items() if k not in ("demand_node", "node_type")}
            built_demands[d["ID"]] = Demand(env=env, demand_node=local[d["demand_node"]], **params)

    return {"nodes": local, "links": built_links, "demands": built_demands,
            "remote_nodes": remote, "outbound_links": outbound, "env": env, "part": part}


# ---- Running the pieces side by side ---------------------------------------
#
# Each piece runs in its own process with its own SimPy environment. Time moves
# in windows as long as the lookahead (the smallest message delay on a cut
# link). Inside a window every piece runs on its own; a message sent during the
# window arrives one delay later, so never before the window ends. At the end
# of each window the pieces stop, swap the messages sent during it, and go on.


class _CutLink(Link):
    """
    A link whose supplier lives in another piece.

    The buyer's two messages (the order, and the "units delivered" notice) leave
    this piece as plain data instead of running code on the supplier, which is
    not here. The parallel runner sets ``piece`` before the run starts.
    """
    piece = None

    def _runner(self):
        if self.piece is None:
            raise RuntimeError(
                f"{self.ID} joins two pieces of a partitioned network. "
                f"Run partitioned networks with run_parallel.")
        return self.piece

    def send_order(self, quantity, accepted, shipped, signal_drop=True):
        self._runner().send_order(self, quantity, accepted, shipped, signal_drop)

    def send_delivered(self, quantity):
        self._runner().send_delivered(self, quantity)


class _ReplyLink:
    """
    Supplier-side stand-in for a cut link, handed to ``Node.serve_order``.

    ``serve_order`` replies with ``link.send(lambda: accepted.succeed(...))``.
    Here ``accepted`` and ``shipped`` are ``_Reply`` objects that post the
    reply to the buyer's piece, with the link's delay, so ``send`` just runs
    the reply at once.
    """
    @staticmethod
    def send(deliver):
        deliver()


class _Reply:
    """Stands in for one of the buyer's events: succeeding it sends the value back to the buyer."""

    def __init__(self, piece, to_part, delay, kind, token):
        self.piece, self.to_part, self.delay, self.kind, self.token = piece, to_part, delay, kind, token

    def succeed(self, value):
        self.piece.post(self.to_part, self.kind, (self.token, value), self.delay)


class _Piece:
    """
    One piece of the network, run one window at a time.

    Messages are tuples ``(arrival_time, send_time, from_piece, number, to_piece,
    kind, payload)``. Sorting them by the first four fields gives the same order
    in every run, so a parallel run is repeatable.
    """

    def __init__(self, nodes, links, demands, assignment, part, sim_time):
        built = build_partition(nodes, links, demands, assignment, part)
        self.env, self.part, self.assignment = built["env"], part, assignment
        self.nodes, self.remote, self.demands = built["nodes"], built["remote_nodes"], built["demands"]
        self.links = {l["ID"]: l for l in links}
        for link in built["links"].values():
            if isinstance(link, _CutLink):
                link.piece = self
        # Nodes here that sell to another piece; their stock level is sent along
        # after each window so the stand-ins over there stay roughly up to date.
        self.sellers = sorted({l["source"] for l in built["outbound_links"]})
        self.outbox = []
        self.count = itertools.count()
        self.waiting = {}  # order number -> the buyer's (accepted, shipped) events
        # The same end marker env.run(until=sim_time) makes in a one-process run,
        # so events at exactly sim_time are left unrun in both.
        self.stop = _deliver_at(self.env, sim_time, lambda: None)

    def post(self, to_part, kind, payload, delay):
        now = self.env.now
        self.outbox.append((now + delay, now, self.part, next(self.count), to_part, kind, payload))

    def send_order(self, link, quantity, accepted, shipped, signal_drop):
        token = next(self.count)
        self.waiting[token] = (accepted, shipped)
        self.post(self.assignment[link.source.ID], "order", (link.ID, quantity, token, signal_drop),
                  link.message_delay)

    def send_delivered(self, link, quantity):
        self.post(self.assignment[link.source.ID], "delivered", (link.ID, quantity), link.message_delay)

    def _receive(self, kind, payload):
        if kind == "order":  # we are the supplier
            link_id, quantity, token, signal_drop = payload
            link = self.links[link_id]
            back, delay = self.assignment[link["sink"]], _delay(link)
            self.env.process(self.nodes[link["source"]].serve_order(
                _ReplyLink, quantity, _Reply(self, back, delay, "answer", token),
                _Reply(self, back, delay, "shipped", token), signal_drop))
        elif kind == "answer":  # we are the buyer
            token, ok = payload
            accepted, _ = self.waiting[token] if ok else self.waiting.pop(token)
            accepted.succeed(ok)
        elif kind == "shipped":
            token, batches = payload
            self.waiting.pop(token)[1].succeed(batches)
        elif kind == "delivered":
            link_id, quantity = payload
            self.nodes[self.links[link_id]["source"]].stats.update_stats(demand_fulfilled=[1, quantity])

    def run_window(self, inbox, levels, until):
        """Take in the messages and stock levels from other pieces, then run every event before ``until``."""
        for node_id, level in levels.items():
            if node_id in self.remote:
                self.remote[node_id].inventory.level = level
        for arrival, _, _, _, _, kind, payload in inbox:
            if arrival < self.env.now:
                raise RuntimeError(
                    f"A message for t={arrival} reached piece {self.part} at t={self.env.now}, after "
                    f"that time had already been simulated. This should not happen; please report it.")
            _deliver_at(self.env, arrival - self.env.now, lambda k=kind, p=payload: self._receive(k, p))
        while self.env.peek() < until:
            self.env.step()
        out, self.outbox = self.outbox, []
        return out, {nid: self.nodes[nid].inventory.level for nid in self.sellers}

    def finish(self):
        """Run to the end; return each node's statistics, then the summaries the network totals need."""
        self.env.run(until=self.stop)
        stats = {nid: node.stats.get_statistics() for nid, node in self.nodes.items()}
        summaries = {nid: _node_summary(node) for nid, node in self.nodes.items()
                     if node.node_type != NodeType.INFINITE_SUPPLIER}
        summaries.update({did: _node_summary(d) for did, d in self.demands.items()})
        return stats, summaries


class _InProcess:
    """Runs a piece in this process (for testing and debugging)."""

    def __init__(self, factory, assignment, part, sim_time):
        self.piece = _Piece(*factory(), assignment, part, sim_time)

    def request(self, method, *args):
        self.result = getattr(self.piece, method)(*args)

    def reply(self):
        return self.result

    def close(self):
        pass


def _worker(conn, factory, assignment, part, sim_time):
    try:
        piece = _Piece(*factory(), assignment, part, sim_time)
        while True:
            method, args = conn.recv()
            conn.send(("ok", getattr(piece, method)(*args)))
            if method == "finish":
                return
    except Exception:
        conn.send(("error", traceback.format_exc()))


class _InWorker:
    """Runs a piece in its own process and talks to it through a pipe."""

    def __init__(self, ctx, factory, assignment, part, sim_time):
        self.part = part
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker, args=(child, factory, assignment, part, sim_time), daemon=True)
        self.proc.start()

    def request(self, method, *args):
        self.conn.send((method, args))

    def reply(self):
        status, value = self.conn.recv()
        if status == "error":
            raise RuntimeError(f"Piece {self.part} failed:\n{value}")
        return value

    def close(self):
        self.proc.join(timeout=5)
        if self.proc.is_alive():
            self.proc.terminate()


def run_parallel(factory, sim_time, n_parts=None, assignment=None, processes=True):
    """
    Run a network split into pieces, each piece in its own process (CPU core).

    The pieces move through time together in steps ("windows") as long as the
    lookahead: the smallest ``message_delay`` on a link between two pieces.
    Within a window each piece runs on its own. After it, the pieces swap the
    messages sent during the window and go on to the next one.

    For a network without randomness the result is the same as running the
    whole network in one process with ``create_sc_net`` and ``env.run`` (same
    message delays). Two cases can differ: messages from different pieces that
    arrive at the very same instant may be handled in another order; and a
    supplier-selection policy that looks at a supplier's stock in another piece
    sees the level from the end of the last window, not the live level.
    Random draws (``random`` in your lambdas, probabilistic disruptions) come
    from each process's own random generator, so random runs match only
    statistically.

    Parameters:
        factory (callable): Function with no arguments that returns
            ``(nodes, links, demands)`` as dicts (the netlist style). It must be
            defined at the top level of an importable module, because every
            process calls it to build its own copy of the network (the lambdas
            in a netlist cannot be sent between processes). It must return the
            same network every time.
        sim_time (float): Time to simulate until.
        n_parts (int, optional): Number of pieces, if ``assignment`` is not given.
        assignment (dict, optional): ``{node_ID: piece}``; by default made with
            ``partition_network``.
        processes (bool): True runs each piece in its own process. False runs
            them one after another in this process (same results; for testing).

    Returns:
        dict: ``stats`` (``{node_ID: statistics dict}`` for every node, as from
        ``node.stats.get_statistics()`` right after the run), ``summaries``
        (what ``simulate_sc_net_parallel`` adds up into network totals),
        ``assignment``, ``lookahead`` (window length) and ``windows`` (number
        of windows run).

    Raises:
        ValueError: If neither ``n_parts`` nor ``assignment`` is given, or the
            split is not valid (see ``check_partition``).
        RuntimeError: If a piece fails; the message holds its traceback.
    """
    nodes, links, demands = factory()
    if assignment is None:
        if n_parts is None:
            raise ValueError("Give n_parts or assignment.")
        assignment = partition_network(nodes, links, demands, n_parts)
    report = check_partition(nodes, links, demands, assignment)
    k, step = report["n_parts"], report["lookahead"]

    pieces = []
    try:
        if processes:
            ctx = multiprocessing.get_context("spawn")  # same behaviour on every OS
            for p in range(k):
                pieces.append(_InWorker(ctx, factory, assignment, p, sim_time))
        else:
            for p in range(k):
                pieces.append(_InProcess(factory, assignment, p, sim_time))
        inbox, levels = [[] for _ in range(k)], {}
        windows, t = 0, 0
        while t < sim_time:
            windows += 1
            t = min(windows * step, sim_time)  # multiply, not add, so window ends don't drift
            for p, piece in enumerate(pieces):
                piece.request("run_window", inbox[p], levels, t)
            inbox, levels = [[] for _ in range(k)], {}
            for piece in pieces:
                out, piece_levels = piece.reply()
                levels.update(piece_levels)
                for message in out:
                    inbox[message[4]].append(message)
            for box in inbox:
                box.sort(key=lambda m: m[:4])
        for piece in pieces:
            piece.request("finish")
        stats, summaries = {}, {}
        for piece in pieces:
            piece_stats, piece_summaries = piece.reply()
            stats.update(piece_stats)
            summaries.update(piece_summaries)
    finally:
        for piece in pieces:
            piece.close()
    return {"stats": stats, "summaries": summaries, "assignment": assignment,
            "lookahead": step, "windows": windows}


def _with_delays(factory, delays):
    """Build the network from ``factory`` and give the links in ``delays`` (``{link_ID: delay}``) that delay."""
    nodes, links, demands = factory()
    links = [dict(link, message_delay=delays[link["ID"]]) if link["ID"] in delays else link for link in links]
    return nodes, links, demands


def simulate_sc_net_parallel(factory, sim_time, n_parts=None, cut_delay=None, assignment=None,
                             processes=True, logging=True):
    """
    Parallel version of ``simulate_sc_net``: split the network into pieces, run
    each piece on its own CPU core, and gather the results.

    Returns the same kind of ``supplychainnet`` dict as ``create_sc_net`` +
    ``simulate_sc_net``: the network totals (``profit``, ``shortage``,
    ``total_demand``, ...) and, under ``"nodes"``, node objects whose ``stats``
    hold the results from the parallel run. (The nodes' other state, such as
    their inventory, is not copied back.) It also has a ``"parallel"`` entry
    describing the split.

    Only links with a ``message_delay`` above 0 can join two pieces. If your
    links have no delay, pass ``cut_delay``: the network is split as if every
    link had that delay, and then only the links that end up between two
    pieces get it. All other links keep their own delay. Note that this
    changes the model a little: orders on those links now take ``cut_delay``
    to reach the supplier, and the answer and shipping notice take
    ``cut_delay`` each to come back.

    Parameters:
        factory (callable): Function with no arguments returning ``(nodes,
            links, demands)`` as dicts, defined at the top level of an
            importable module (see ``run_parallel``). ``functools.partial`` of
            such a function also works, e.g. ``partial(build_network, 500)``.
        sim_time (float): Time to simulate until.
        n_parts (int, optional): Number of pieces (usually the number of CPU
            cores to use). Needed unless ``assignment`` is given.
        cut_delay (float, optional): Message delay to give the links that are
            cut and have none.
        assignment (dict, optional): ``{node_ID: piece}``, to choose the split
            yourself.
        processes (bool): False runs the pieces one after another in this
            process (same results; for testing).
        logging (bool): Write the network totals to the log, as
            ``simulate_sc_net`` does. The pieces themselves run without logging.

    Returns:
        dict: The ``supplychainnet`` dict described above. ``"parallel"`` holds
        ``assignment``, ``cut_links``, ``lookahead`` and ``windows``.

    Raises:
        ValueError: If ``cut_delay`` is not above 0, if neither ``n_parts`` nor
            ``assignment`` is given, or if the split is not valid.
    """
    nodes, links, demands = factory()
    if cut_delay is not None:
        if not cut_delay > 0:
            raise ValueError("cut_delay must be above 0.")
        if assignment is None:
            if n_parts is None:
                raise ValueError("Give n_parts or assignment.")
            every_link = [dict(link, message_delay=_delay(link) or cut_delay) for link in links]
            assignment = partition_network(nodes, every_link, demands, n_parts)
        delays = {link["ID"]: cut_delay for link in links
                  if not _delay(link) and assignment[link["source"]] != assignment[link["sink"]]}
        factory = functools.partial(_with_delays, factory, delays)

    result = run_parallel(factory, sim_time, n_parts=n_parts, assignment=assignment, processes=processes)

    # Rebuild the whole network here (it is not run) so the results can be put
    # in the same dict, with the same node objects, as a one-process run gives.
    if logging:
        global_logger.enable_logging()
    else:
        global_logger.disable_logging()
    nodes, links, demands = factory()
    supplychainnet = create_sc_net(nodes, links, demands)
    for nid, node in supplychainnet["nodes"].items():
        stats = result["summaries"][nid]["stats"] if nid in result["summaries"] else result["stats"][nid]
        for key, value in stats.items():
            setattr(node.stats, key, value)
    for did, demand in supplychainnet["demands"].items():
        for key, value in result["summaries"][did]["stats"].items():
            setattr(demand.stats, key, value)
    _fill_network_kpis(
        supplychainnet,
        [result["summaries"][n["ID"]] for n in nodes if n["ID"] in result["summaries"]],
        [result["summaries"][d["ID"]] for d in demands])
    supplychainnet["parallel"] = {
        "assignment": result["assignment"],
        "cut_links": check_partition(nodes, links, demands, result["assignment"])["cut_links"],
        "lookahead": result["lookahead"],
        "windows": result["windows"],
    }
    _log_network_kpis(supplychainnet)
    return supplychainnet
