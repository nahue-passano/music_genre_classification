import torch
from core.trainer_system import Trainer
from models.cnn import CNN
from data.dataset import train_loader, valid_loader


# Model instantiation
model = CNN()

# Loss and optimizer  instantiation
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam
# TODO: Emprolijar como se pasa loss_function y optimizer al trainer
# no me parece prolijo que por un lado pase una instancia y por otro
# pase la clase

# Trainer instantiation
trainer = Trainer(
    model=model,
    train_set=train_loader,
    valid_set=valid_loader,
    loss_function=loss_function,
    optimizer=optimizer,
)

trainer.fit()
