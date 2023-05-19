import torch
from tqdm import tqdm
from models.cnn import CNN
from data.dataset import train_loader, valid_loader, test_loader

# TODO: Agregar en el constructor un atributo para los hiperparámetros (lea un dict)

# Hyperparameters
EPOCHS = 10
LEARNING_RATE = 1e-3


class Trainer:
    def __init__(self, model, train_set, valid_set, loss_function, optimizer):
        self.model = model
        self.train_set = train_set
        self.valid_set = valid_set
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_hyperparameters(self, epochs=EPOCHS, learning_rate=LEARNING_RATE):
        """_summary_

        Parameters
        ----------
        epochs : _type_, optional
            _description_, by default EPOCHS
        learning_rate : _type_, optional
            _description_, by default LEARNING_RATE
        """
        self.epochs = epochs
        self.learning_rate = learning_rate

    def _set_optimizer(self):
        # Optimizer setting
        self.optimizer = self.optimizer(
            params=self.model.parameters(), lr=self.learning_rate
        )

    def _train_batch_processing(self):
        train_loss = 0.0  # Cumulative loss in epoch_i
        train_acc = 0.0  # Cumulative accuracy in epoch_i
        self.model.train()

        for inputs, targets in self.train_set:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            # 1. Batch forward pass
            probabilities = self.model(inputs)
            # 2. Loss per batch
            loss = self.loss_function(probabilities, targets)
            train_loss += loss.item()
            # 3. Initialize optimizer gradients in 0
            self.optimizer.zero_grad()
            # 4. Loss backward
            loss.backward()
            # 5. Optimizer step
            self.optimizer.step()
            # 6. Calculating accuracy
            predictions = torch.argmax(probabilities, dim=1)
            train_acc += sum(predictions == targets).item()

        # Mean of the train loss and accuracy counting number of train batches
        train_loss /= len(self.train_set)
        train_acc /= len(self.train_set)

        return train_loss, train_acc

    def _valid_batch_processing(self):
        valid_loss = 0.0
        valid_acc = 0.0

        self.model.eval()
        with torch.no_grad():
            # Validation loop
            for valid_inputs, valid_targets in self.valid_set:
                valid_inputs = valid_inputs.to(self.device)
                valid_targets = valid_targets.to(self.device)
                # 1. Forward pass
                valid_probabilities = self.model(valid_inputs)
                # 2. Loss per validation batch
                valid_loss += self.loss_function(valid_probabilities, valid_targets)
                # 3. Calculating accuracy
                predictions = torch.argmax(valid_probabilities, dim=1)
                valid_acc += sum(predictions == valid_targets).item()

            # Mean of the validation loss and accuracy counting number of valid batches
            valid_loss /= len(self.valid_set)
            valid_acc /= len(self.valid_set)

        return valid_loss, valid_acc

    def fit(self):
        # Loading model into device
        self.model.to(self.device)

        # Set train hyperparameters
        self.set_hyperparameters()

        # Set optimizer
        self._set_optimizer()

        for epoch_i in tqdm(range(self.epochs)):
            print(f"\n >>> Epoch {epoch_i+1}: ")
            # Train batch processing
            train_loss, train_acc = self._train_batch_processing()

            # Validation batch processing
            valid_loss, valid_acc = self._valid_batch_processing()

            # Loss and accuracy results
            print(
                f"Train loss = {train_loss:.4f}"
                f" | Test loss= {valid_loss:.4f}"
                "   ---   "
                f"Train accuracy = {train_acc:.4f}"
                f" | Test accuracy= {valid_acc:.4f}"
            )


if __name__ == "__main__":
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

    # # Saving model
    # model_name = 'model.pt'
    # torch.save(model.state_dict(), ROOT / model_name)
