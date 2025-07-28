from torch.optim import AdamW
import copy
import gc
from core.datasets import *
import argparse
from core.models import *
from core.utils import *

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    base_dirs = ["../datasets/gqa","../datasets/art"]
    task_ids = [0,1]
    label2ids = {}
    vocabs= {}
    num_labels = {}
    for tid in task_ids:
        vocab, label2id, _ = load_mappings(base_dirs[tid])
        label2ids[tid]=label2id
        vocabs[tid]=vocab
        num_labels[tid]=len(vocab)
        
    # 1) load processor & model
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    base_model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
    model = ViltWithCustomClassifier(base_model)
    
    add_adapters_to_vilt(model, bottleneck=256)

    model.to(device)
    datasets = {}
    train_sizes = {}
    test_sizes = {}
    train_loaders = {}
    test_loaders = {}

    num_dirichlet_clusters = args.num_dirichlet_clusters
    n_users               = args.n_users
    alpha                 = args.alpha
    is_test               = False
    epochs                = args.epochs
    cons_rounds           = args.cons_rounds
    users_per_cluster     = args.users_per_cluster
    n_clusters            = int(n_users/users_per_cluster)
    gaf                   = args.gaf
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    accuracy_record = []
    path = f"results/hierarchy_{gaf}_{users_per_cluster}"
    os.makedirs(path, exist_ok=True)
    save_dir = f"hierarchy_{gaf}_{users_per_cluster}"

    clients_per_cluster = [n_users]
    partitioner = {}
    user_loaders = {}
    user_test_loaders = {}

    for tid in task_ids:
        annotation_path = os.path.join(base_dirs[tid], "ann.json")
        dataset_path = os.path.join(base_dirs[tid],"train")
        if tid==0:
            datasets[tid]=LocalGQADataset(annotation_path,dataset_path, label2ids[tid])
        elif tid==1:
            datasets[tid]=LocalArtDataset(annotation_path,dataset_path, label2ids[tid])
        elif tid==2:
            datasets[tid]=LocalVizWizDataset(annotation_path,dataset_path, label2ids[tid])

        # Split into 90% train / 10% test
        train_sizes[tid]=int(0.9 * len(datasets[tid]))
        test_sizes[tid]=len(datasets[tid]) - train_sizes[tid]


        train_ds, test_ds= torch.utils.data.random_split(datasets[tid], [train_sizes[tid], test_sizes[tid]], generator=torch.Generator().manual_seed(SEED))

        partitioner[tid] = NonIIDPartition(dataset=train_ds, test_data=test_ds,num_clients_per_cluster=clients_per_cluster, tid = tid,
                                      num_clusters=num_dirichlet_clusters, alpha=alpha, processor=processor, batch_size=8)
        user_loaders[tid] = partitioner[tid].client_loaders

        base = len(test_ds) // n_users
        sizes = [base + (1 if i<len(test_ds) % n_users else 0) for i in range(n_users)]
        shards = torch.utils.data.random_split(test_ds, sizes, generator=torch.Generator().manual_seed(SEED))

        if torch.cuda.is_available():
            test_loaders[tid]=DataLoader(test_ds, batch_size=8, shuffle=False,  num_workers =2, collate_fn=lambda b: collate_fn(b, processor))
            user_test_loaders[tid]=[ DataLoader(shard, batch_size=8, shuffle=True, num_workers=2, collate_fn=lambda b, proc=processor: collate_fn(b, proc)) for shard in shards]
        else:
            test_loaders[tid]=DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=lambda b: collate_fn(b, processor))
            user_test_loaders[tid]=[ DataLoader(shard, batch_size=8, shuffle=True, collate_fn=lambda b, proc=processor: collate_fn(b, proc)) for shard in shards]


        

    # 6) optimizer & loss
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    clusters = {cid: [users_per_cluster * cid + i for i in range(users_per_cluster)] for cid in range(n_clusters)}
    print(clusters)

    classifier_modules = {}
    for tid, nl in num_labels.items():
        main_head =nn.Linear(model.hidden_size, nl)
        for uid in range(n_users):
            classifier_modules[(tid, uid)]=copy.deepcopy(main_head)
        
    personalized_adapters = {cid: [{k: v.detach().cpu().clone() for k, v in layer.output.adapter.state_dict().items()} for layer in model.vilt.encoder.layer] for cid in clusters.keys()}
   
    for ep in range(1, epochs+1):
        print(f"\n--- Global Epoch {ep} ---")
        avg_acc = [0,0]

        adapter_accum_global = {k: torch.zeros_like(v, device="cpu") for k, v in model.state_dict().items() if "adapter" in k}
        adapter_count_global = 0
        # 1b) Head accumulators (one per task)
        head_accum_global  = {tid: { k: torch.zeros_like(v, device="cpu") for k, v in classifier_modules[(tid,0)].state_dict().items() } for tid in task_ids}
        head_count_global = {tid: 0 for tid in task_ids}

        for cid, users in clusters.items():
            adapter_accum = {k: torch.zeros_like(v, device="cpu") for k, v in model.state_dict().items() if "adapter" in k}
            adapter_count = 0
            # 1b) Head accumulators (one per task)
            head_accum  = {tid: { k: torch.zeros_like(v, device="cpu") for k, v in classifier_modules[(tid,0)].state_dict().items() } for tid in task_ids}
            head_count = {tid: 0 for tid in task_ids}
            for uid in users:
                model.classifier=[None, None]
                local_model = model
                local_adapters =[{ k: v.clone() for k, v in layer_state.items() } for layer_state in personalized_adapters[cid]]
                load_adapters_for_task(local_model,local_adapters)

                for tid in task_ids:
                    
                    head = classifier_modules[(tid, uid)]
                    local_model.set_classifier(head, tid)

                    trainable = list(local_model.classifier[tid].parameters()) + [p for n,p in local_model.vilt.named_parameters() if "adapter" in n]
                    optim = AdamW(trainable, lr=1e-4)
                    total_loss = 0.0
                    local_model.train()
                    # loop = tqdm(user_loaders[tid][uid], desc=f"Epoch {ep}, Task {tid}, User {uid} ", unit="batch")
                    N_i = len(user_loaders[tid][uid].dataset)
                    if is_test:
                        loop = tqdm(user_loaders[tid][uid], desc=f"Epoch {ep}, Task {tid}, User {uid} ", unit="batch")
                    else:
                        loop = user_loaders[tid][uid]
                    total_examples = 0
                    for batch in loop:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        labels = batch.pop("labels")
                        outputs = local_model.forward_task(tid,**batch)
                        logits  = outputs
                        loss    = criterion(logits, labels)
                        loss.backward()
                        optim.step()
                        optim.zero_grad()
                        total_loss += loss.item()
                    
                    print(f"{total_loss}:{len(user_loaders[tid][uid])}")
                    print(f"[Epoch {ep}], Cluster {cid}, Task {tid}, User {uid}, train loss = {total_loss/len(user_loaders[tid][uid]):.4f}")
                    acc = evaluate(local_model, user_test_loaders[tid][uid],tid, ep, is_test)
                    avg_acc[tid]+=acc

                    accuracy_record.append({
                                "epoch":    ep,
                                "cluster":  cid,
                                "user":     uid,
                                "task":     tid,
                                "loss":     total_loss/len(user_loaders[tid][uid]),
                                "accuracy": acc
                            })
                
                    with open(f"{path}/accuracy_record.json", "w") as f:
                        json.dump(accuracy_record, f, indent=4)

                    sd = local_model.state_dict()

                    # a) adapters
                    for k in adapter_accum:
                        adapter_accum[k] += sd[k].detach().cpu()*N_i
                    adapter_count += N_i
                    

                    # b) this task’s head
                    hsd = head.state_dict()
                    for k in head_accum[tid]:
                        head_accum[tid][k] += hsd[k].detach().cpu()*N_i
                    head_count[tid] += N_i

                    if ep%gaf == 0:
                        for k in adapter_accum_global:
                            adapter_accum_global[k] += adapter_accum[k]
                        adapter_count_global += adapter_count
                        
                        for k in head_accum_global[tid]:
                            head_accum_global[tid][k] += head_accum[tid][k]
                        head_count_global[tid] += head_count[tid]


                    del optim      
                    gc.collect()    
                    if torch.cuda.is_available():           
                        torch.cuda.empty_cache()
            print(f"Local aggregation of cluster {cid}...")
            for tid in task_ids:
                # Personalized adapter aggregation
                for layer_idx, layer_state in enumerate(personalized_adapters[cid]):
                    new_layer_state = {}
                    for param_name in layer_state.keys():
                        full_key = f"vilt.encoder.layer.{layer_idx}.output.adapter.{param_name}"
                        avg = adapter_accum[full_key] / adapter_count
                        new_layer_state[param_name] = avg.clone()
                    
                    personalized_adapters[cid][layer_idx] = new_layer_state

                # Classification head
                cnt = head_count[tid]
                new_head_sd = {}
                for k, tot in head_accum[tid].items():
                    new_head_sd[k] = (tot / cnt).to(device)
                for uid in users:
                    classifier_modules[(tid,uid)].load_state_dict(new_head_sd)

        for tid in task_ids:
            print(f"[Epoch {ep}], avg test accuracy of task {tid} = {avg_acc[tid]/n_users:.2f}%\n")
        
        if ep%gaf == 0:
            print(f"Global aggregation...")
            for tid in task_ids:
                # Personalized adapter aggregation
                for layer_idx, layer_state in enumerate(personalized_adapters[0]):
                    new_layer_state = {}
                    for param_name in layer_state.keys():
                        full_key = f"vilt.encoder.layer.{layer_idx}.output.adapter.{param_name}"
                        avg = adapter_accum_global[full_key] / adapter_count_global
                        new_layer_state[param_name] = avg.clone()
                    for cid in clusters.keys():
                        personalized_adapters[cid][layer_idx] = new_layer_state

                # Classification head
                cnt = head_count_global[tid]
                new_head_sd = {}
                for k, tot in head_accum_global[tid].items():
                    new_head_sd[k] = (tot / cnt).to(device)
                for cid, users in clusters.items():
                    for uid in users:
                        classifier_modules[(tid,uid)].load_state_dict(new_head_sd)
        
        save_model_components(classifier_modules, personalized_adapters,  save_dir = save_dir)



def execute():
    parser = argparse.ArgumentParser(description="Run federated VILT with adapters")
    parser.add_argument("--num_dirichlet_clusters", type=int,   default=1)
    parser.add_argument("--n_users",               type=int,   default=40)
    parser.add_argument("--alpha",                 type=float, default=0.1)
    parser.add_argument("--is_test",               action="store_true")
    parser.add_argument("--epochs",                type=int,   default=200)
    parser.add_argument("--cons_rounds",           type=int,   default=100)
    parser.add_argument("--n_clusters",            type=int,   default=10)
    parser.add_argument("--users_per_cluster",     type=int,   default=4)
    parser.add_argument("--gaf",                   type=int,   default=2)
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    execute()
