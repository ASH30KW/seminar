import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, hparams, input_size=28 * 28, latent_dim=20):
        super().__init__()

        # set hyperparams
        self.latent_dim = latent_dim
        self.input_size = input_size
        self.hparams = hparams
        self.encoder = None

        ########################################################################
        # TODO: Initialize your encoder!                                       #
        #                                                                      #
        # Possible layers: nn.Linear(), nn.BatchNorm1d(), nn.ReLU(),           #
        # nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU().                             #
        # Look online for the APIs.                                            #
        #                                                                      #
        # Hint 1:                                                              #
        # Wrap them up in nn.Sequential().                                     #
        # Example: nn.Sequential(nn.Linear(10, 20), nn.ReLU())                 #
        #                                                                      #
        # Hint 2:                                                              #
        # The latent_dim should be the output size of your encoder.            #
        # We will have a closer look at this parameter later in the exercise.  #
        ########################################################################

        self.encoder = nn.Sequential(
            nn.Linear(input_size, self.hparams["n_hidden_1"]),
            nn.ReLU(),
            nn.BatchNorm1d(self.hparams["n_hidden_1"]),
            nn.Linear(self.hparams["n_hidden_1"], self.hparams["n_hidden_2"]),
            nn.ReLU(),
            nn.BatchNorm1d(self.hparams["n_hidden_2"]),
            nn.Linear(self.hparams["n_hidden_2"], self.hparams["n_hidden_3"]),
            nn.ReLU(),
            nn.BatchNorm1d(self.hparams["n_hidden_3"]),
            nn.Linear(self.hparams["n_hidden_3"], self.hparams["latent_size"]),
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # feed x into encoder!
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, hparams, latent_dim=20, output_size=28 * 28):
        super().__init__()

        # set hyperparams
        self.hparams = hparams
        self.decoder = None

        ########################################################################
        # TODO: Initialize your decoder!                                       #
        ########################################################################

        self.decoder = nn.Sequential(
            nn.Linear(self.hparams["latent_size"], self.hparams["n_hidden_3"]),
            nn.ReLU(),
            nn.BatchNorm1d(self.hparams["n_hidden_3"]),
            nn.Linear(self.hparams["n_hidden_3"], self.hparams["n_hidden_2"]),
            nn.ReLU(),
            nn.BatchNorm1d(self.hparams["n_hidden_2"]),
            nn.Linear(self.hparams["n_hidden_2"], self.hparams["n_hidden_1"]),
            nn.ReLU(),
            nn.BatchNorm1d(self.hparams["n_hidden_1"]),
            nn.Linear(self.hparams["n_hidden_1"], output_size),
            nn.Sigmoid(),  # Ensure input data is normalized between 0 and 1
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # feed x into decoder!
        return self.decoder(x)


class Autoencoder(nn.Module):
    def __init__(self, hparams, encoder, decoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        # Define models
        self.encoder = encoder
        self.decoder = decoder
        self.device = hparams.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.set_optimizer()

    def forward(self, x):
        reconstruction = None
        ########################################################################
        # TODO: Feed the input image to your encoder to generate the latent    #
        #  vector. Then decode the latent vector and get your reconstruction   #
        #  of the input.                                                       #
        ########################################################################

        # Encode the input image
        latent_vector = self.encoder(x)

        # Decode the latent vector to get the reconstruction
        reconstruction = self.decoder(latent_vector)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return reconstruction

    def set_optimizer(self):
        self.optimizer = None
        ########################################################################
        # TODO: Define your optimizer.                                         #
        ########################################################################

        # Define the optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(), self.hparams["learning_rate"]
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def training_step(self, batch, loss_func):
        images = batch[0].to(self.device)
        
        # Check if batch size is 1 and expand
        if images.shape[0] == 1:
            images = images.repeat(2, 1, 1, 1)  # Repeat the batch to make size 2
        
        images = images.view(images.shape[0], -1)
        self.optimizer.zero_grad()
        self.train()
        encoded = self.encoder(images)
        decoded = self.decoder(encoded)
        loss = loss_func(decoded, images)
        loss.backward()
        self.optimizer.step()
        return loss

    def validation_step(self, batch, loss_func):
        """
        This function is called for every batch of data during validation.
        It should return the loss for the batch.
        """
        loss = None
        ########################################################################
        # TODO:                                                                #
        # Complete the validation step, similraly to the way it is shown in    #
        # train_classifier() in the notebook.                                  #
        #                                                                      #
        # Hint 1:                                                              #
        # Here we don't supply as many tips. Make sure you follow the pipeline #
        # from the notebook.                                                   #
        ########################################################################

        # Extract the input batch
        images = batch[0].to(self.device)

        if images.shape[0] == 1:
            images = images.repeat(2, 1, 1, 1)  # Repeat the batch to make size 2
                
        images = images.view(images.shape[0], -1)

        # Forward pass: Encode and decode
        encoded = self.encoder(images)
        decoded = self.decoder(encoded)

        # Compute the loss between the input and the reconstruction
        loss = loss_func(decoded, images)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return loss

    def getReconstructions(self, loader=None):
        assert loader is not None, "Please provide a dataloader for reconstruction"
        self.eval()
        self = self.to(self.device)

        reconstructions = []

        for batch in loader:
            X = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            reconstruction = self.forward(flattened_X)
            reconstructions.append(
                reconstruction.view(-1, 28, 28).cpu().detach().numpy()
            )

        return np.concatenate(reconstructions, axis=0)


class Classifier(nn.Module):
    def __init__(self, hparams, encoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        self.encoder = encoder
        self.model = nn.Identity()
        self.device = hparams.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        ########################################################################
        # TODO:                                                                #
        # Given an Encoder, finalize your classifier, by adding a classifier   #
        # block of fully connected layers.                                     #
        ########################################################################

        self.model = nn.Sequential(
            nn.Linear(self.hparams["latent_size"], self.hparams["n_hidden_4"]),
            nn.ReLU(),
            nn.Linear(self.hparams["n_hidden_4"], self.hparams["num_classes"])
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        self.set_optimizer()

    def forward(self, x):
        x = self.encoder(x)
        x = self.model(x)
        return x

    def set_optimizer(self):
        self.optimizer = None
        ########################################################################
        # TODO: Implement your optimizer. Send it to the classifier parameters #
        # and the relevant learning rate (from self.hparams)                   #
        ########################################################################

        self.optimizer = torch.optim.Adam(
            self.parameters(), self.hparams["learning_rate"]
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def getAcc(self, loader=None):
        assert loader is not None, "Please provide a dataloader for accuracy evaluation"

        self.eval()
        self = self.to(self.device)

        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            score = self.forward(flattened_X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc
