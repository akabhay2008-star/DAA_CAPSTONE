# Mini-Courier Planner - DAA Capstone Assignment (ETCCDA302)
# Abhay Kumar | Roll No: 2501730185 | B.Tech CSE (Section A)

import csv
import heapq
import os
import random
import time
from itertools import combinations

INF = float("inf")


# ---------------- Phase 1: Data & Graph ----------------

class Parcel:
    def __init__(self, parcel_id, destination, weight, value, deadline, priority):
        self.parcel_id = parcel_id
        self.destination = destination
        self.weight = int(weight)
        self.value = int(value)
        self.deadline = int(deadline)
        self.priority = int(priority)


class Graph:
    def __init__(self):
        self.adj = {}

    def add_node(self, node):
        if node not in self.adj:
            self.adj[node] = []

    def add_edge(self, u, v, dist):
        if dist < 0:
            raise ValueError("Distance cannot be negative.")
        self.add_node(u)
        self.add_node(v)
        self.adj[u].append((v, dist))
        self.adj[v].append((u, dist))

    def num_edges(self):
        return sum(len(v) for v in self.adj.values()) // 2


SMALL_EDGES = [
    ("Depot", "A", 4), ("Depot", "B", 7), ("Depot", "C", 6), ("Depot", "D", 13),
    ("A", "B", 3), ("A", "D", 8), ("B", "C", 2), ("B", "E", 6),
    ("C", "F", 5), ("D", "E", 4), ("E", "G", 3), ("E", "F", 9), ("F", "G", 7),
]

SMALL_PARCELS = [
    ("P01", "A", 5, 40, 4, 3), ("P02", "B", 8, 55, 6, 3), ("P03", "C", 3, 20, 10, 1),
    ("P04", "D", 6, 35, 8, 2), ("P05", "A", 4, 25, 12, 2), ("P06", "E", 9, 60, 5, 3),
    ("P07", "F", 2, 15, 14, 1), ("P08", "G", 7, 30, 10, 2),
    ("P09", "C", 5, 38, 7, 2), ("P10", "B", 3, 18, 16, 1),
]


def get_small_dataset():
    parcels = [Parcel(*row) for row in SMALL_PARCELS]
    g = Graph()
    for u, v, d in SMALL_EDGES:
        g.add_edge(u, v, d)
    return parcels, g, 20


def generate_graph(n_nodes, seed=1):
    rng = random.Random(seed)
    names = ["Depot"] + [f"C{i}" for i in range(1, n_nodes)]
    g = Graph()
    seen = set()
    for i in range(1, n_nodes):
        j = rng.randrange(i)
        g.add_edge(names[i], names[j], rng.randint(2, 25))
        seen.add(frozenset((i, j)))
    extra = int(1.5 * n_nodes)
    while extra > 0:
        i, j = rng.sample(range(n_nodes), 2)
        if frozenset((i, j)) not in seen:
            seen.add(frozenset((i, j)))
            g.add_edge(names[i], names[j], rng.randint(2, 25))
            extra -= 1
    return g


def generate_parcels(count, destinations, seed=1):
    rng = random.Random(seed)
    parcels = []
    for i in range(1, count + 1):
        prio = rng.choice([1, 2, 3])
        parcels.append(Parcel(
            f"P{i:04d}", rng.choice(destinations), rng.randint(1, 12),
            prio * 10 + rng.randint(0, 25), rng.randint(2, max(4, 8 + count // 10)), prio
        ))
    return parcels


def get_medium_dataset():
    g = generate_graph(25, seed=11)
    return generate_parcels(60, [f"C{i}" for i in range(1, 25)], seed=11), g, 50


def get_large_dataset():
    g = generate_graph(100, seed=12)
    return generate_parcels(500, [f"C{i}" for i in range(1, 100)], seed=12), g, 50


def load_csv(parcels_file, graph_file):
    parcels = []
    with open(parcels_file, newline="") as f:
        for r in csv.DictReader(f):
            parcels.append(Parcel(
                r["parcel_id"].strip(), r["destination"].strip(),
                r["weight_kg"], r["value"], r["deadline_h"], r["priority"]
            ))
    g = Graph()
    with open(graph_file, newline="") as f:
        for r in csv.DictReader(f):
            d = float(r["distance_km"])
            g.add_edge(r["from"].strip(), r["to"].strip(), int(d) if d.is_integer() else d)
    return parcels, g


# ---------------- Phase 2: Sorting & 0/1 Knapsack ----------------

def sort_key(p):
    return (p.deadline, -p.priority, p.parcel_id)


def merge_sort(arr):
    if len(arr) <= 1:
        return list(arr)
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    res, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if sort_key(left[i]) <= sort_key(right[j]):
            res.append(left[i])
            i += 1
        else:
            res.append(right[j])
            j += 1
    return res + left[i:] + right[j:]


def quick_sort(arr):
    if len(arr) <= 1:
        return list(arr)
    pivot = sort_key(arr[random.randrange(len(arr))])
    less = [x for x in arr if sort_key(x) < pivot]
    same = [x for x in arr if sort_key(x) == pivot]
    more = [x for x in arr if sort_key(x) > pivot]
    return quick_sort(less) + same + quick_sort(more)


def insertion_sort(arr):
    a = list(arr)
    for i in range(1, len(a)):
        curr = a[i]
        j = i - 1
        while j >= 0 and sort_key(a[j]) > sort_key(curr):
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = curr
    return a


def knapsack_dp(parcels, capacity):
    n = len(parcels)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        w, v = parcels[i - 1].weight, parcels[i - 1].value
        for c in range(capacity + 1):
            dp[i][c] = dp[i - 1][c]
            if w <= c and dp[i - 1][c - w] + v > dp[i][c]:
                dp[i][c] = dp[i - 1][c - w] + v
    chosen, c = [], capacity
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            chosen.append(parcels[i - 1])
            c -= parcels[i - 1].weight
    chosen.reverse()
    return dp[n][capacity], chosen, dp


def print_dp_table(parcels, dp, capacity):
    print(" " * 15 + "".join(f"{c:>4}" for c in range(capacity + 1)))
    print(f"{'0 (none)':<15}" + "".join(f"{x:>4}" for x in dp[0]))
    for i, p in enumerate(parcels, 1):
        label = f"{p.parcel_id} w{p.weight} v{p.value}"
        print(f"{label:<15}" + "".join(f"{x:>4}" for x in dp[i]))


# ---------------- Phase 3: Dijkstra & Nearest-Neighbour Route ----------------

def dijkstra(graph, source):
    dist = {node: INF for node in graph.adj}
    parent = {}
    dist[source] = 0
    heap = [(0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in graph.adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                parent[v] = u
                heapq.heappush(heap, (d + w, v))
    return dist, parent


def get_path(parent, src, dst):
    if src == dst:
        return [src]
    if dst not in parent:
        return []
    path = [dst]
    while path[-1] != src:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def nearest_neighbour_route(graph, depot, stops):
    cache = {}

    def shortest(node):
        if node not in cache:
            cache[node] = dijkstra(graph, node)
        return cache[node]

    unvisited = sorted(set(stops) - {depot})
    curr, order, legs, total = depot, [], [], 0
    while unvisited:
        dist, parent = shortest(curr)
        reachable = [s for s in unvisited if dist[s] < INF]
        if not reachable:
            break
        nxt = min(reachable, key=lambda s: (dist[s], s))
        legs.append((curr, nxt, dist[nxt], get_path(parent, curr, nxt)))
        total += dist[nxt]
        order.append(nxt)
        unvisited.remove(nxt)
        curr = nxt
    if order:
        dist, parent = shortest(curr)
        legs.append((curr, depot, dist[depot], get_path(parent, curr, depot)))
        total += dist[depot]
    return order, legs, total


# ---------------- Full Planner Pipeline ----------------

def run_planner(name, parcels, graph, capacity, depot="Depot", speed=30.0, max_trips=None):
    t0 = time.perf_counter()
    sorted_parcels = merge_sort(parcels)
    dist_depot, _ = dijkstra(graph, depot)
    remaining, undeliverable = [], []
    for p in sorted_parcels:
        if p.weight <= 0 or p.value < 0:
            undeliverable.append((p, "invalid data (weight must be > 0, value >= 0)"))
        elif p.weight > capacity:
            undeliverable.append((p, "overweight"))
        elif dist_depot.get(p.destination, INF) == INF:
            undeliverable.append((p, "unreachable"))
        else:
            remaining.append(p)

    trips, clock = [], 0.0
    while remaining:
        dp_in = list(remaining)
        val, chosen, dp = knapsack_dp(remaining, capacity)
        if not chosen:
            # Only zero-value parcels are left - the DP never selects them, so
            # report them instead of silently dropping them.
            for p in remaining:
                undeliverable.append((p, "zero value - never selected by knapsack"))
            remaining = []
            break
        order, legs, total_km = nearest_neighbour_route(
            graph, depot, [p.destination for p in chosen]
        )
        arrivals, d_sum = {}, 0.0
        for _, dst, dist, _ in legs:
            d_sum += dist
            arrivals[dst] = clock + d_sum / speed
        end_time = clock + total_km / speed
        late = [p for p in chosen if arrivals.get(p.destination, clock) > p.deadline]
        trips.append((chosen, val, order, legs, total_km, clock, end_time, arrivals, late, dp, dp_in))
        done = {p.parcel_id for p in chosen}
        remaining = [p for p in remaining if p.parcel_id not in done]
        clock = end_time

    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"\n=== MINI-COURIER PLANNER ({name.upper()}) ===")
    print(f"Parcels: {len(parcels)} | Nodes: {len(graph.adj)} | Roads: {graph.num_edges()} | Capacity: {capacity} kg")
    print("\nSTEP 1: Sorted Parcels (Merge Sort)")
    print(f"{'ID':<7}{'Dest':<7}{'Kg':>4}{'Val':>6}{'Due':>6}{'Prio':>6}")
    for p in sorted_parcels[:15]:
        print(f"{p.parcel_id:<7}{p.destination:<7}{p.weight:>4}{p.value:>6}{p.deadline:>5}h{p.priority:>6}")
    if len(sorted_parcels) > 15:
        print(f"... ({len(sorted_parcels) - 15} more)")

    show = trips if max_trips is None else trips[:max_trips]
    for idx, (chosen, val, order, legs, km, st, et, arr, late, dp, dp_in) in enumerate(show, 1):
        print(f"\n--- TRIP {idx} ---")
        if idx == 1 and len(dp_in) <= 15 and capacity <= 30:
            print_dp_table(dp_in, dp, capacity)
        print("Loaded:", ", ".join(p.parcel_id for p in chosen),
              f"| {sum(p.weight for p in chosen)}/{capacity} kg | Value: {val}")
        print("Route:", " -> ".join([depot] + order + [depot]))
        for src, dst, d, path in legs:
            print(f"  {src:>6} -> {dst:<6} {d:>4g} km via {' > '.join(path)}")
        print(f"Distance: {km:g} km ({st:.2f}h -> {et:.2f}h)")
        if late:
            print("Late:", ", ".join(f"{p.parcel_id} (due {p.deadline}h, arr {arr[p.destination]:.1f}h)" for p in late))
        else:
            print("All parcels arrived on time.")
    if len(show) < len(trips):
        print(f"\n... ({len(trips) - len(show)} more trips hidden)")

    delivered = sum(len(t[0]) for t in trips)
    total_late = sum(len(t[8]) for t in trips)
    total_km = sum(t[4] for t in trips)
    total_val = sum(t[1] for t in trips)
    print(f"\nSUMMARY: Trips={len(trips)} | Delivered={delivered}/{len(parcels)} | "
          f"Undeliverable={len(undeliverable)} | Late={total_late} | "
          f"Distance={total_km:g} km | Value={total_val} | Time={elapsed_ms:.2f} ms")
    for p, reason in undeliverable[:10]:
        print(f"  Undeliverable {p.parcel_id} -> {p.destination}: {reason}")
    if len(undeliverable) > 10:
        print(f"  ... ({len(undeliverable) - 10} more undeliverable)")
    return {"trips": len(trips), "delivered": delivered, "undeliverable": len(undeliverable),
            "late": total_late, "km": total_km, "value": total_val, "ms": elapsed_ms}


# ---------------- Phase 4: Benchmarks, Charts & Tests ----------------

def timeit(fn, repeat=3):
    best = INF
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def run_benchmarks():
    print("\n=== PERFORMANCE BENCHMARKS (ms) ===")
    sort_sizes = [100, 250, 500, 1000, 2000]
    m_times, q_times, i_times, b_times = [], [], [], []
    print("\n[1] Sorting")
    print(f"{'n':>6}{'Merge':>10}{'Quick':>10}{'Insertion':>11}{'Built-in':>10}")
    for n in sort_sizes:
        d = generate_parcels(n, ["A", "B", "C"], seed=n)
        tm = timeit(lambda: merge_sort(d))
        tq = timeit(lambda: quick_sort(d))
        ti = timeit(lambda: insertion_sort(d), 1)
        tb = timeit(lambda: sorted(d, key=sort_key))
        m_times.append(tm); q_times.append(tq); i_times.append(ti); b_times.append(tb)
        print(f"{n:>6}{tm*1000:>10.3f}{tq*1000:>10.3f}{ti*1000:>11.3f}{tb*1000:>10.3f}")

    knap_sizes = [50, 100, 200, 400, 800]
    knap_times = []
    print("\n[2] Knapsack DP (W=100)")
    for n in knap_sizes:
        d = generate_parcels(n, ["A"], seed=n)
        t = timeit(lambda: knapsack_dp(d, 100))
        knap_times.append(t)
        print(f"  n={n:<4} -> {t*1000:.3f} ms")

    v_sizes = [100, 200, 400, 800, 1600]
    dij_times = []
    print("\n[3] Dijkstra")
    for v in v_sizes:
        g = generate_graph(v, seed=v)
        t = timeit(lambda: dijkstra(g, "Depot"))
        dij_times.append(t)
        print(f"  V={v:<5} -> {t*1000:.3f} ms")

    results = {"sort_sizes": sort_sizes, "merge": m_times, "quick": q_times,
               "insertion": i_times, "builtin": b_times,
               "knap_sizes": knap_sizes, "knap": knap_times,
               "dij_sizes": v_sizes, "dij": dij_times}
    try:
        import matplotlib
        if os.environ.get("NO_GUI"):
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs("outputs", exist_ok=True)
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
        ax[0].plot(sort_sizes, i_times, marker="o", label="Insertion O(n^2)")
        ax[0].plot(sort_sizes, q_times, marker="o", label="Quick Sort")
        ax[0].plot(sort_sizes, m_times, marker="o", label="Merge O(n log n)")
        ax[0].plot(sort_sizes, b_times, marker="o", label="Python sorted()")
        ax[0].set(title="Sorting (log scale)", xlabel="Parcels (n)", ylabel="Seconds")
        ax[0].set_yscale("log")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)

        ax[1].plot(knap_sizes, knap_times, marker="o", color="green")
        ax[1].set(title="Knapsack DP (W=100)", xlabel="Parcels (n)", ylabel="Seconds")
        ax[1].grid(alpha=0.3)

        ax[2].plot(v_sizes, dij_times, marker="o", color="red")
        ax[2].set(title="Dijkstra (Binary Heap)", xlabel="Nodes (V)", ylabel="Seconds")
        ax[2].grid(alpha=0.3)

        fig.suptitle("Mini-Courier Planner - Runtime Benchmarks")
        fig.tight_layout()
        fig.savefig("outputs/runtime_charts.png", dpi=130)
        print("\nChart saved to outputs/runtime_charts.png")
        plt.close(fig)
    except ImportError:
        print("\nmatplotlib is not installed. Run: pip install matplotlib")
    return results


def run_tests():
    sample = generate_parcels(100, ["A", "B", "C"], seed=3)
    exp = sorted(sample, key=sort_key)
    assert merge_sort(sample) == exp and quick_sort(sample) == exp and insertion_sort(sample) == exp

    items = generate_parcels(10, ["A"], seed=1)
    best, chosen, _ = knapsack_dp(items, 20)
    brute = max(sum(p.value for p in c) for r in range(11) for c in combinations(items, r)
                if sum(p.weight for p in c) <= 20)
    assert best == brute and sum(p.value for p in chosen) == best

    g = Graph()
    g.add_edge("Depot", "A", 4)
    g.add_edge("A", "D", 8)
    g.add_edge("Depot", "D", 13)
    dist, parent = dijkstra(g, "Depot")
    assert dist["D"] == 12 and get_path(parent, "Depot", "D") == ["Depot", "A", "D"]

    # Edge cases: zero-value, overweight and unreachable parcels must be reported
    import contextlib
    import io
    eg = Graph()
    eg.add_edge("Depot", "A", 4)
    eg.add_node("Island")
    edge_parcels = [
        Parcel("Z1", "A", 3, 0, 5, 1),
        Parcel("H1", "A", 50, 10, 5, 1),
        Parcel("U1", "Island", 2, 9, 5, 1),
        Parcel("OK", "A", 2, 9, 5, 1),
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        res = run_planner("edge", edge_parcels, eg, 10)
    assert res["delivered"] == 1 and res["undeliverable"] == 3
    print("\nAll correctness tests passed!")


# ---------------- Main Menu ----------------

def main():
    while True:
        print("\n--- MINI-COURIER PLANNER ---")
        print("1. Small Dataset   2. Medium Dataset   3. Large Dataset")
        print("4. Custom CSV      5. Benchmarks & Graphs   6. Run Tests   0. Exit")
        ch = input("Choice: ").strip()
        if ch == "1":
            run_planner("small", *get_small_dataset())
        elif ch == "2":
            run_planner("medium", *get_medium_dataset())
        elif ch == "3":
            run_planner("large", *get_large_dataset(), max_trips=3)
        elif ch == "4":
            p_file = input("Parcels CSV [sample_data/parcels.csv]: ").strip() or "sample_data/parcels.csv"
            g_file = input("Graph CSV [sample_data/graph.csv]: ").strip() or "sample_data/graph.csv"
            try:
                cap = int(input("Capacity (kg) [20]: ").strip() or "20")
                depot = input("Depot node name [Depot]: ").strip() or "Depot"
                parcels, graph = load_csv(p_file, g_file)
                if depot not in graph.adj:
                    print(f"Depot '{depot}' is not a node in the graph.")
                else:
                    run_planner("custom", parcels, graph, cap, depot=depot)
            except (OSError, KeyError, ValueError) as err:
                print(f"Could not load data: {err!r}")
        elif ch == "5":
            run_benchmarks()
        elif ch == "6":
            run_tests()
        elif ch == "0":
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
