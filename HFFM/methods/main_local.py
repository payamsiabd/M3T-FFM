
import copy
import gc
from torch.optim import AdamW
import argparse
from core.datasets import *
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
    num_dirichlet_clusters = args.num_dirichlet_clusters
    n_users               = args.n_users
    alpha                 = args.alpha
    is_test               = args.is_test
    epochs                = args.epochs
    cons_rounds           = args.cons_rounds
    users_per_cluster     = args.users_per_cluster
    n_clusters            = int(n_users/users_per_cluster)
    gaf                   = args.gaf
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    accuracy_record = []
    path = f"results/local"
    os.makedirs(path, exist_ok=True)

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
            user_test_loaders[tid]=[DataLoader(shard, batch_size=8, shuffle=True, num_workers=2, collate_fn=lambda b, proc=processor: collate_fn(b, proc)) for shard in shards]
        else:
            user_test_loaders[tid]=[DataLoader(shard, batch_size=8, shuffle=True, collate_fn=lambda b, proc=processor: collate_fn(b, proc)) for shard in shards]

    # 6) optimizer & loss
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)


    classifier_modules = {}
    for tid, nl in num_labels.items():
        main_head =nn.Linear(model.hidden_size, nl)
        for uid in range(n_users):
            classifier_modules[(tid, uid)]=copy.deepcopy(main_head)
        
    personalized_adapters = {tid: [[{ k: v.detach().cpu().clone() for k, v in layer.output.adapter.state_dict().items()} for layer in model.vilt.encoder.layer ] for uid in range(n_users)] for tid in task_ids}
    
    for ep in range(1, epochs+1):
        print(f"\n--- Global Epoch {ep} ---")
        avg_acc = [0,0]
        for uid in range(n_users):
            model.classifier=[None, None]
            local_model = model

            load_adapters_for_task(local_model, personalized_adapters[0][uid])

            for tid in task_ids:
                
                head = classifier_modules[(tid, uid)]
                local_model.set_classifier(head, tid)

                trainable = list(local_model.classifier[tid].parameters()) + [p for n,p in local_model.vilt.named_parameters() if "adapter" in n]
                optim = AdamW(trainable, lr=1e-4)
                total_loss = 0.0
                local_model.train()

                if is_test:
                    loop = tqdm(user_loaders[tid][uid], desc=f"Epoch {ep}, Task {tid}, User {uid} ", unit="batch")
                else:
                    loop = user_loaders[tid][uid]
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

                print(f"[Epoch {ep}], Task {tid}, User {uid}, train loss = {total_loss/len(user_loaders[tid][uid]):.4f}")
                acc = evaluate(local_model, user_test_loaders[tid][uid],tid, ep, is_test)
                avg_acc[tid]+=acc

                accuracy_record.append({
                                "epoch":    ep,
                                "cluster":  0,
                                "user":     uid,
                                "task":     tid,
                                "loss":     total_loss/len(user_loaders[tid][uid]),
                                "accuracy": acc
                            })
                
                with open(f"{path}/accuracy_record.json", "w") as f:
                    json.dump(accuracy_record, f, indent=4)

                del optim
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        for tid in task_ids:
            print(f"[Epoch {ep}], avg test accuracy of task {tid} = {avg_acc[tid]/n_users:.2f}%\n")
        
        print('saving...')
        save_model_components(classifier_modules, personalized_adapters)



def execute():
    parser = argparse.ArgumentParser(description="Run federated VILT with adapters")
    parser.add_argument("--num_dirichlet_clusters", type=int,   default=1)
    parser.add_argument("--n_users",               type=int,   default=40)
    parser.add_argument("--alpha",                 type=float, default=0.1)
    parser.add_argument("--is_test",               action="store_true")
    parser.add_argument("--epochs",                type=int,   default=50)
    parser.add_argument("--cons_rounds",           type=int,   default=100)
    parser.add_argument("--n_clusters",            type=int,   default=10)
    parser.add_argument("--users_per_cluster",     type=int,   default=4)
    parser.add_argument("--gaf",                   type=int,   default=5)
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    execute()

