import numpy as np
import networkx as nx
import random
from itertools import combinations
import pandas as pd
import os

from core.network import calculate_transmission_rate

SEED = 49
global_random_gn=np.random.default_rng(SEED)



def main():
    # === PARAMETERS ===
    transmit_power = 200  # Mw
    n_servers = 10
    n_users = 40
    min_dist = 500.0  # minimum edge‐server separation (m)
    area_size = 5000.0  # make sure area is large enough to fit 20 servers ≥1000m apart
    model_size = 6 * 8000  # 1 MB = 8000 kbit

    # --- 1) Generate edge‐server positions with minimum‐distance constraint ---
    servers_positions = []
    while len(servers_positions) < n_servers:
        x, y = global_random_gn.uniform(0, area_size, size=2)
        if all(np.hypot(x - sx, y - sy) >= min_dist for sx, sy in servers_positions):
            servers_positions.append((x, y))
    servers_positions = np.array(servers_positions)

    user_max_dist_to_server = 1000.0
    user_cluster_radius = 500.0
    users_per_cluster = 4

    users_positions = []
    clusters = {cid: [] for cid in range(n_servers)}
    uid = -1

    for sid, (sx, sy) in enumerate(servers_positions):
        # pick a cluster center within 500 m of the server
        while True:
            r_center = np.sqrt(global_random_gn.random()) * user_max_dist_to_server
            theta_center = global_random_gn.random() * 2 * np.pi
            cx = sx + r_center * np.cos(theta_center)
            cy = sy + r_center * np.sin(theta_center)
            if 0 <= cx <= area_size and 0 <= cy <= area_size:
                break

        # generate users tightly clustered around (cx, cy)
        for _ in range(users_per_cluster):
            uid += 1
            while True:
                r_user = np.sqrt(global_random_gn.random()) * user_cluster_radius
                theta_user = global_random_gn.random() * 2 * np.pi
                ux = cx + r_user * np.cos(theta_user)
                uy = cy + r_user * np.sin(theta_user)
                if 0 <= ux <= area_size and 0 <= uy <= area_size:
                    users_positions.append((ux, uy))
                    clusters[sid].append(uid)
                    break

    users_positions = np.array(users_positions)
    path  = 'clusters.npy'
    np.save(path,clusters)
    # clusters = np.array(clusters)

    up_data_rates = []
    for cid, users in clusters.items():
        for uid in users:
            up_data_rates.append(calculate_transmission_rate(users_positions[uid], servers_positions[cid], 1000))
            ux, uy = users_positions[uid]
            sx, sy = servers_positions[cid]
            print(f"Data rate: {up_data_rates[uid]:.2f} --> User {uid} at ({ux:.0f}, {uy:.0f}) → Server {cid} at ({sx:.0f}, {sy:.0f})")

    cluster_pair_rates = {cid: {} for cid in clusters}
    cluster_pair_latencies = {cid: {} for cid in clusters}
    for cid, uids in clusters.items():
        for i, uid1 in enumerate(uids):
            for uid2 in uids[i + 1:]:
                # positions of the two users
                p1 = users_positions[uid1]
                p2 = users_positions[uid2]
                # compute the transmission rate between them
                rate = calculate_transmission_rate(p1, p2, transmit_power, bandwidth=1000)
                # store it keyed by the (uid1, uid2) tuple
                cluster_pair_rates[cid][(uid1, uid2)] = rate
                cluster_pair_latencies[cid][(uid1, uid2)] = model_size/ rate

    # --- example: print them out ---
    for cid, rates in cluster_pair_rates.items():
        print(f"\nCluster {cid} pairwise data‐rates (kb/s):")
        for (u1, u2), r in rates.items():
            print(f"  Users {u1}↔{u2}: {r:.2f}")

    up_latencies = {uid: 0 for uid in range(n_users)}
    cloud_latencies = {uid: 0 for uid in range(n_users)}

    server_to_cloud_latencies = {cid: 0 for cid in clusters.keys()}

    for cid, users in clusters.items():
        temp= 0
        for uid in users:
            latency = model_size / up_data_rates[uid]
            up_latencies[uid] = latency
            # cloud_latencies[uid] = 10* latency
            cloud_latencies[uid] = 0.1 + model_size / 4000
            temp += 0.1 + model_size / 4000

        temp/=len(users)
        server_to_cloud_latencies[cid] = temp

    # Scenario 1
    epochs=50
    gaf = 2
    cloud_transmit_power = 800
    local_transmit_power = 600

    avg_fedavg_latencies ={ep:  max([cloud_latencies[uid] for cid, users in clusters.items() for uid in users])  for ep in range(epochs)}
    avg_fedavg_energies ={ep: sum([cloud_latencies[uid]*cloud_transmit_power for cid, users in clusters.items() for uid in users])  for ep in range(epochs)}


    # Scenario 2
    local_latencies = {cid: max([up_latencies[uid]*gaf for uid in users]) for cid, users in clusters.items()}
    avg_hierarchy_latencies_2_4 = {ep: max([(server_to_cloud_latencies[cid]+local_latencies[cid])/gaf for cid, users in clusters.items()])  for ep in range(epochs)}

    local_energies = {cid: sum([up_latencies[uid] * gaf*local_transmit_power for uid in users]) for cid, users in clusters.items()}
    avg_hierarchy_energy_2_4 = {ep: sum([(server_to_cloud_latencies[cid]*cloud_transmit_power + local_energies[cid])/gaf for cid, users in clusters.items()])  for ep in range(epochs)}


    # Scenario 4
    gaf = 8
    local_latencies = {cid: max([up_latencies[uid] * gaf for uid in users]) for cid, users in clusters.items()}
    avg_hierarchy_latencies_8_4 = { ep: max([(server_to_cloud_latencies[cid] + local_latencies[cid]) / gaf for cid, users in clusters.items()]) for ep in range(epochs)}

    local_energies = {cid: sum([up_latencies[uid] * gaf * local_transmit_power for uid in users]) for cid, users in  clusters.items()}
    avg_hierarchy_energy_8_4 = {ep: sum([(server_to_cloud_latencies[cid] * cloud_transmit_power + local_energies[cid]) / gaf for cid, users in  clusters.items()]) for ep in range(epochs)}

    # Scenario 5
    gaf = 200
    local_latencies = {cid: max([up_latencies[uid] * gaf for uid in users]) for cid, users in clusters.items()}
    avg_hierarchy_latencies_200_4 = {ep: max([(server_to_cloud_latencies[cid] + local_latencies[cid]) / gaf for cid, users in clusters.items()]) for ep in range(epochs)}

    local_energies = {cid: sum([up_latencies[uid] * gaf * local_transmit_power for uid in users]) for cid, users in clusters.items()}
    avg_hierarchy_energy_200_4 = {ep: sum([(local_energies[cid]) / gaf for cid, users in clusters.items()]) for ep in range(epochs)}

    # Scenario 3

    cluster_pair_latencies = {cid: {} for cid in clusters}
    for cid, uids in clusters.items():
        for u1 in uids:
            for u2 in uids:
                if u1 == u2:
                    continue
                p1, p2 = users_positions[u1], users_positions[u2]
                rate = calculate_transmission_rate(p1, p2, transmit_power, bandwidth=8000)  # kb/s
                lat = model_size / rate  # s
                cluster_pair_latencies[cid][(u1, u2)] = lat

    # connectivity threshold
    conn_radius = 30

    relay_latencies_up = {}
    relay_latencies_down = {}
    relay_energies_up = {}
    relay_energies_down = {}
    d2d_local_transmit_power=600
    for cid, uids in clusters.items():
        agg = random.choice(uids)
        print(f"\n=== Cluster {cid}: Aggregator = User {agg} ===")

        # 1) Build directed geometric graph
        G = nx.DiGraph()
        G.add_nodes_from(uids)
        for u1, u2 in combinations(uids, 2):
            p1, p2 = users_positions[u1], users_positions[u2]
            if np.linalg.norm(p1 - p2) <= conn_radius:
                # if within range, add both directed edges
                G.add_edge(u1, u2, weight=cluster_pair_latencies[cid][(u1, u2)])
                G.add_edge(u2, u1, weight=cluster_pair_latencies[cid][(u2, u1)])

        # 2) Ensure strong connectivity by patching via undirected MST
        if not nx.is_strongly_connected(G):
            # build an undirected surrogate with min(lat,lat_rev)
            H = nx.Graph()
            H.add_nodes_from(uids)
            for (u1, u2), lat in cluster_pair_latencies[cid].items():
                lat_rev = cluster_pair_latencies[cid].get((u2, u1), np.inf)
                H.add_edge(u1, u2, weight=min(lat, lat_rev))
            T = nx.minimum_spanning_tree(H, weight='weight')
            for u1, u2, d in T.edges(data=True):
                if not G.has_edge(u1, u2):
                    G.add_edge(u1, u2, weight=cluster_pair_latencies[cid][(u1, u2)])
                if not G.has_edge(u2, u1):
                    G.add_edge(u2, u1, weight=cluster_pair_latencies[cid][(u2, u1)])

        # 4) Shortest paths & latencies: user → agg
        fpaths, flats = {}, {}
        for uid in uids:
            if uid == agg:
                fpaths[uid], flats[uid] = [agg], 0.0
            else:
                fpaths[uid] = nx.shortest_path(G, uid, agg, weight='weight')
                flats[uid] = nx.shortest_path_length(G, uid, agg, weight='weight')
            print(f"User {uid} → Agg {agg} | Path {fpaths[uid]} | Latency {flats[uid]:.3f}s")

        # 5) Energy to aggregate: sum P·t over all users → agg, convert to kJ
        energy_to_agg_kJ = {uid: d2d_local_transmit_power * flats[uid] for uid in uids}
        total_e2a = sum(energy_to_agg_kJ.values())
        print(f"\nCluster {cid}: Energy to aggregate = {total_e2a:.4f} kJ")

        # 6) Shortest paths: agg → each user (directed)
        bpaths, blats = {}, {}
        for uid in uids:
            if uid == agg:
                bpaths[uid], blats[uid] = [agg], 0.0
            else:
                bpaths[uid] = nx.shortest_path(G, agg, uid, weight='weight')
                blats[uid] = nx.shortest_path_length(G, agg, uid, weight='weight')
            print(f"Agg {agg} → User {uid} | Path {bpaths[uid]} | Latency {blats[uid]:.3f}s")

        # 7) Compute broadcast tree edges (unique) and energy
        broadcast_edges = set()
        for path in bpaths.values():
            for i in range(len(path) - 1):
                broadcast_edges.add((path[i], path[i + 1]))

        energy_bcast_kJ = 0.0
        for u, v in broadcast_edges:
            lat = G[u][v]['weight']
            e = d2d_local_transmit_power * lat
            energy_bcast_kJ += e
            print(f"  Edge {u}→{v}: Lat {lat:.3f}s → E {e:.6f} kJ")

        bcast_time = max(blats.values())
        print(f"\nCluster {cid}: Broadcast time = {bcast_time:.3f}s")
        print(f"Cluster {cid}: Energy to broadcast = {energy_bcast_kJ:.4f} kJ")

        # 8) Total round‑trip energy
        print(f"Cluster {cid}: Total RT energy = {total_e2a + energy_bcast_kJ:.4f} kJ\n")

        relay_latencies_up[cid] =max(flats.values())
        relay_latencies_down[cid] = bcast_time
        relay_energies_up[cid] = total_e2a
        relay_energies_down[cid] = energy_bcast_kJ

    gaf = 2
    local_latencies = {cid: np.mean([up_latencies[uid] for uid in users]) for cid, users in clusters.items()}
    avg_relay_latencies_2_4 ={ep: max([(server_to_cloud_latencies[cid] + (relay_latencies_up[cid]+relay_latencies_down[cid]) * (gaf-1) + relay_latencies_up[cid]+local_latencies[cid])/gaf
                                    for cid, users in clusters.items()])  for ep in range(epochs)}

    avg_relay_energy_2_4 ={ep: sum([(server_to_cloud_latencies[cid]*cloud_transmit_power + (
                relay_energies_up[cid] + relay_energies_down[cid]) * (gaf - 1) + relay_energies_up[cid]+local_latencies[cid]*local_transmit_power)/gaf for cid, users in clusters.items()]) for ep in range(epochs)}


    gaf = 8
    local_latencies = {cid: np.mean([up_latencies[uid] for uid in users]) for cid, users in clusters.items()}
    avg_relay_latencies_8_4 = {ep: max([(server_to_cloud_latencies[cid] + (
                relay_latencies_up[cid] + relay_latencies_down[cid]) * (gaf - 1) + relay_latencies_up[cid] +
                                         local_latencies[cid]) / gaf
                                        for cid, users in clusters.items()]) for ep in range(epochs)}

    avg_relay_energy_8_4 = {ep: sum([(server_to_cloud_latencies[cid] * cloud_transmit_power + (
            relay_energies_up[cid] + relay_energies_down[cid]) * (gaf - 1) + relay_energies_up[cid] + local_latencies[
                                          cid] * local_transmit_power) / gaf for cid, users in clusters.items()]) for ep in
                            range(epochs)}

    gaf = 200
    local_latencies = {cid: np.mean([up_latencies[uid] for uid in users]) for cid, users in clusters.items()}
    avg_relay_latencies_200_4 = {ep: max([(server_to_cloud_latencies[cid] + (
                relay_latencies_up[cid] + relay_latencies_down[cid]) * (gaf - 1) + relay_latencies_up[cid] +
                                         local_latencies[cid]) / gaf
                                        for cid, users in clusters.items()]) for ep in range(epochs)}

    avg_relay_energy_200_4 = {ep: sum([(( relay_energies_up[cid] + relay_energies_down[cid]) * (gaf - 1) + relay_energies_up[cid] + local_latencies[
                                          cid] * local_transmit_power) / gaf for cid, users in clusters.items()]) for ep in
                            range(epochs)}



    def load_metrics_df(folder_name):
        path = os.path.join("results", folder_name, "accuracy_record.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return pd.read_json(path)

    df_fedavg = load_metrics_df("hierarchy_1_4")

    df_hierarchy_2_4 = load_metrics_df("hierarchy_2_4")

    df_hierarchy_8_4 = load_metrics_df("hierarchy_8_4")

    df_hierarchy_200_4 = load_metrics_df("hierarchy_200_4")


    def mean_from_epoch_dict(d):
        return np.mean(list(d.values()))

    # Last common epoch
    common_epochs = set(df_fedavg["epoch"]).intersection(
        df_hierarchy_2_4["epoch"],
        df_hierarchy_8_4["epoch"],
        df_hierarchy_200_4["epoch"]
    )
    last_common_epoch = max(common_epochs)

    loss_fedavg_last = df_fedavg[df_fedavg["epoch"] == last_common_epoch]["accuracy"].mean()
    loss_hierarchy_2_last = df_hierarchy_2_4[df_hierarchy_2_4["epoch"] == last_common_epoch]["accuracy"].mean()

    loss_hierarchy_8_last = df_hierarchy_8_4[df_hierarchy_8_4["epoch"] == last_common_epoch]["accuracy"].mean()

    loss_hierarchy_200_last = df_hierarchy_200_4[df_hierarchy_200_4["epoch"] == last_common_epoch]["accuracy"].mean()

    results = {
        "Method": ["Star Topology", "HFFM 2", "HFFM 8", "Edge Only", "HFFM+D2D 2", "HFFM+D2D 8", "Edge Only + D2D"],
        "Avg Latency (s)": [
            mean_from_epoch_dict(avg_fedavg_latencies),
            mean_from_epoch_dict(avg_hierarchy_latencies_2_4),
            mean_from_epoch_dict(avg_hierarchy_latencies_8_4),
            mean_from_epoch_dict(avg_hierarchy_latencies_200_4),
            mean_from_epoch_dict(avg_relay_latencies_2_4),
            mean_from_epoch_dict(avg_relay_latencies_8_4),
            mean_from_epoch_dict(avg_relay_latencies_200_4)
        ],
        "Avg Energy (J)": [
            mean_from_epoch_dict(avg_fedavg_energies)/1000,
            mean_from_epoch_dict(avg_hierarchy_energy_2_4)/1000,
            mean_from_epoch_dict(avg_hierarchy_energy_8_4) / 1000,
            mean_from_epoch_dict(avg_hierarchy_energy_200_4) / 1000,
            mean_from_epoch_dict(avg_relay_energy_2_4)/1000,
            mean_from_epoch_dict(avg_relay_energy_8_4) / 1000,
            mean_from_epoch_dict(avg_relay_energy_200_4) / 1000
        ],
        "Loss at Last Epoch": [
            loss_fedavg_last,
            loss_hierarchy_2_last,
            loss_hierarchy_8_last,
            loss_hierarchy_200_last,
            loss_hierarchy_2_last,
            loss_hierarchy_8_last,
            loss_hierarchy_200_last
        ]
    }

    df = pd.DataFrame(results)
    print(df.to_latex(
        index=False,
        float_format="%.4f",
        caption=f"Comparison at Last Common Epoch {last_common_epoch}",
        label="tab:comparison"
    ))


if __name__ == "__main__":
    main()
