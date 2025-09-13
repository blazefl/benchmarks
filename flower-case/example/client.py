from typing import OrderedDict
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from .task import load_data
from .models import get_model
from example.task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = get_model(str(context.run_config["model-name"]), num_classes=10)
    arrays = msg.content["arrays"]
    assert isinstance(arrays, ArrayRecord)
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    batch_size = int(context.run_config["batch-size"])
    trainloader = load_data(partition_id, num_partitions, batch_size)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Construct and return reply Message
    model_state_dict = model.state_dict()
    assert isinstance(model_state_dict, OrderedDict)
    model_record = ArrayRecord(model_state_dict)
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),  # pyright: ignore[reportArgumentType]
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)
