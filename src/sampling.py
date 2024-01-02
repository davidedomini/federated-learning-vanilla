import numpy as np
from torchvision import datasets, transforms

def mnist_noniid_unequal(dataset, num_users):
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shards = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    min_shards_per_client = 1
    max_shards_per_client = 30

    random_shard_size = np.random.randint(
        min_shards_per_client,
        max_shards_per_client+1,
        size=num_users
    )
    random_shard_size = np.around(random_shard_size / sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    if sum(random_shard_size) > num_shards:
        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has at least one shard of data
            rand_set = set(np.random.choice(idx_shards, 1, replace=False))
            idx_shards = list(set(idx_shards) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs: (rand+1)*num_imgs]),
                    axis=0
                )
        random_shard_size = random_shard_size - 1

        for i in range(num_users):
        # Next, randomly assign the remaining shards
            if len(idx_shards) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shards):
                shard_size = len(idx_shards)
            rand_set = set(np.random.choice(idx_shards, 1, replace=False))
            idx_shards = list(set(idx_shards) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]),
                    axis=0
                )
    else:
        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shards, 1, replace=False))
            idx_shards = list(set(idx_shards) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]),
                    axis=0
                )

            if len(idx_shards) > 0:
                shard_size = len(idx_shards)
                k = min(dict_users, key=lambda x: len(dict_users.get(x)))
                rand_set = set(np.random.choice(idx_shards, 1, replace=False))
                idx_shards = list(set(idx_shards) - rand_set)
                for rand in rand_set:
                    dict_users[i] = np.concatenate(
                        (dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]),
                        axis=0
                    )

    return dict_users
