from ray import tune

search_space = {
    "lr" : tune.uniform(1e-5, 1e-3)
}
