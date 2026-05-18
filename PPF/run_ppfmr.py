import argparse
import os
import time

import numpy as np
import torch

import data_utils
import ppfmr
import utils


def parse_seed_list(seed_text: str):
    vals = [x.strip() for x in seed_text.split(",") if len(x.strip()) > 0]
    return [int(v) for v in vals]


def default_num_pred(dataset: str):
    d = dataset.lower()
    if d == "facebook":
        return 200
    if d == "lj":
        return 1000
    if d in ["amazon", "dblp", "twitter"]:
        return 5000
    if d in ["cmnee-pr", "cmnee_pr", "cmnee"]:
        return 2500
    return 2500


if __name__ == "__main__":
    print("= " * 20)
    print("## Starting Time:", utils.get_cur_time(), flush=True)

    parser = argparse.ArgumentParser(description="PPF-MR: Pattern-Perception and Propagation-Feature Fusion")

    parser.add_argument("--dataset", type=str, default="amazon")
    parser.add_argument("--data_root", type=str, default="../KDD2024ProCom-master/data")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")

    parser.add_argument("--from_scratch", type=int, default=1)
    parser.add_argument("--save_ckpt", type=int, default=1)
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")

    parser.add_argument("--gnn_type", type=str, default="GCN")
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)

    parser.add_argument("--pretrain_epoch", type=int, default=30)
    parser.add_argument("--pretrain_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--walk_len", type=int, default=128)
    parser.add_argument("--restart_prob", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--lambda_struct", type=float, default=1.0)

    parser.add_argument("--run_times", type=int, default=10)
    parser.add_argument("--num_shot", type=int, default=10)
    parser.add_argument("--num_pred", type=int, default=-1)

    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--k_hop", type=int, default=2)
    parser.add_argument("--tau_scale", type=float, default=0.0)
    parser.add_argument("--max_candidate_size", type=int, default=40)
    parser.add_argument("--scope_cap", type=int, default=300)
    parser.add_argument("--assignment_rounds", type=int, default=2)

    parser.add_argument("--refine_epoch", type=int, default=30)
    parser.add_argument("--refine_threshold", type=float, default=0.5)
    parser.add_argument("--neg_ratio", type=float, default=1.0)
    parser.add_argument("--candidate_batch_size", type=int, default=64)

    parser.add_argument("--wo_node_consistency", action="store_true")
    parser.add_argument("--wo_struct_consistency", action="store_true")
    parser.add_argument("--wo_propagation_feature", action="store_true")
    parser.add_argument("--wo_refinement", action="store_true")

    args = parser.parse_args()

    if args.num_pred <= 0:
        args.num_pred = default_num_pred(args.dataset)

    seed_list = parse_seed_list(args.seeds)
    if len(seed_list) == 0:
        seed_list = [0]

    if args.wo_node_consistency and args.wo_struct_consistency:
        raise ValueError("Both node consistency and structure consistency are disabled, please keep at least one.")

    if args.device.startswith("cuda") and (not torch.cuda.is_available()):
        print(f"CUDA unavailable, switch {args.device} -> cpu")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(args)
    print("\n")

    num_node, num_edge, num_comm, graph_data, nx_graph, communities = data_utils.load_dataset(
        data_root=args.data_root,
        dataset=args.dataset,
    )
    print(f"Finish loading data: {graph_data}\n")

    graph_struct = ppfmr.GraphStruct.from_nx(nx_graph, num_nodes=num_node)

    model = ppfmr.PPFMR(
        input_dim=graph_data.x.size(1),
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        gnn_type=args.gnn_type,
        device=device,
        temperature=args.temperature,
        lambda_struct=args.lambda_struct,
    )

    num_param = sum(p.numel() for p in model.encoder.parameters())
    print(f"[Parameters] Number of parameters in Encoder: {num_param}")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_file = os.path.join(
        args.ckpt_dir,
        f"ppfmr_{args.dataset}_{args.gnn_type}_{args.n_layers}_{args.hidden_dim}.pt",
    )

    utils.set_seed(seed_list[0])
    if args.from_scratch == 0 and os.path.exists(ckpt_file):
        model.encoder.load_state_dict(torch.load(ckpt_file, map_location=device))
        print(f"Load pre-trained encoder from {ckpt_file}\n")
    else:
        print("Start pattern-perception pretraining ...")
        st = time.time()
        model.fit_pattern_encoder(
            graph_data=graph_data,
            graph_struct=graph_struct,
            epochs=args.pretrain_epoch,
            batch_size=args.pretrain_batch_size,
            lr=args.lr,
            walk_len=args.walk_len,
            restart_prob=args.restart_prob,
            use_node_consistency=not args.wo_node_consistency,
            use_struct_consistency=not args.wo_struct_consistency,
        )
        print(f"Pretraining finished, cost {time.time() - st:.3f}s\n")

        if args.save_ckpt:
            torch.save(model.encoder.state_dict(), ckpt_file)
            print(f"Save pre-trained encoder to {ckpt_file}\n")

    node_emb = model.get_node_embeddings(graph_data).detach().cpu()

    print("Generating propagation-aware initial candidates ...")
    st = time.time()
    initial_candidates = ppfmr.generate_candidates(
        node_emb=node_emb,
        struct=graph_struct,
        num_candidates=args.num_pred,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        seed_hops=args.k_hop,
        tau_scale=args.tau_scale,
        max_candidate_size=args.max_candidate_size,
        scope_cap=args.scope_cap,
        assignment_rounds=args.assignment_rounds,
        use_propagation_feature=not args.wo_propagation_feature,
    )
    print(f"Initial candidates #{len(initial_candidates)}, cost {time.time() - st:.3f}s\n")

    fixed_candidate_emb = None
    if args.wo_refinement:
        fixed_candidate_emb = model.encode_communities(
            initial_candidates,
            graph_data,
            batch_size=args.candidate_batch_size,
        ).numpy()

    all_scores = []
    for i in range(args.run_times):
        print(f"Times {i}")

        seed = seed_list[i % len(seed_list)]
        utils.set_seed(seed)

        idx = list(range(num_comm))
        np.random.shuffle(idx)

        if args.num_shot >= num_comm:
            train_idx = idx[: max(1, num_comm - 1)]
            test_idx = idx[max(1, num_comm - 1):]
        else:
            train_idx = idx[: args.num_shot]
            test_idx = idx[args.num_shot:]

        train_comms = [communities[j] for j in train_idx]
        test_comms = [communities[j] for j in test_idx]

        if len(test_comms) == 0:
            print("No test communities left after split, skip this run.")
            continue

        if args.wo_refinement:
            work_candidates = initial_candidates
            candidate_emb = fixed_candidate_emb
        else:
            refiner = ppfmr.NodeRetentionMLP(args.hidden_dim).to(device)
            ppfmr.train_refiner(
                refiner=refiner,
                node_emb=node_emb,
                train_comms=train_comms,
                struct=graph_struct,
                lr=args.lr,
                epochs=args.refine_epoch,
                neg_ratio=args.neg_ratio,
                extend_hops=args.k_hop,
            )
            work_candidates = ppfmr.refine_candidate_set(
                candidates=initial_candidates,
                refiner=refiner,
                node_emb=node_emb,
                threshold=args.refine_threshold,
            )
            print(f"Refined candidates #{len(work_candidates)}")

            candidate_emb = model.encode_communities(
                work_candidates,
                graph_data,
                batch_size=args.candidate_batch_size,
            ).numpy()
            del refiner

        train_emb = model.encode_communities(
            train_comms,
            graph_data,
            batch_size=args.candidate_batch_size,
        ).numpy()

        pred_comms = ppfmr.match_candidates(
            train_emb=train_emb,
            candidate_emb=candidate_emb,
            candidate_comms=work_candidates,
            num_pred=args.num_pred,
        )

        f1, jaccard = ppfmr.evaluate_predictions(pred_comms, test_comms, verbose=True)
        print(f"Run-{i}: F1={f1:.4f}, Jaccard={jaccard:.4f}")
        utils.pred_community_analysis(pred_comms)

        all_scores.append([f1, jaccard])
        print("\n")

    if len(all_scores) == 0:
        print("No valid runs were executed.")
    else:
        scores = np.asarray(all_scores)
        avg = scores.mean(axis=0)
        std = scores.std(axis=0)

        print(f"Overall F1 {avg[0]:.4f}+-{std[0]:.5f}")
        print(f"Overall Jaccard {avg[1]:.4f}+-{std[1]:.5f}")

    print("\n## Finishing Time:", utils.get_cur_time(), flush=True)
    print("= " * 20)
    print("Done!")
