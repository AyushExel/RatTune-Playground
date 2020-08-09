from ray import tune
from ray.tune.integration.wandb import wandb_mixin
from ray.util.sgd.torch import TrainingOperator
from ray.util.sgd.utils import override


import wandb

@wandb_mixin
def train_fn(config):
    for i in range(10):
        loss = 100
        wandb.log({"loss": loss})
    tune.report(loss=loss, done=True)


tune.run(
    train_fn,
    config={
        # define search space here
        "a": tune.choice([1, 2, 3]),
        "b": tune.choice([4, 5, 6]),
        # wandb configuration
        "wandb": {
            "project": "Optimization_Project",
             "api_key": "80436e2f16d020c8f91e3ef6e2f29a10efb84f9b"
        }
    })