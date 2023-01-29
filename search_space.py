import ray

search_space = {
    "lr" : ray.tune.uniform(1e-5, 1e-3)
}
