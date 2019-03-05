import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from model import Encoder, Decoder
from torchvision import transforms
from tensorboardX import SummaryWriter
from torchvision import datasets
import torch.distributions as D
import torch.nn.functional as F



#weight initialization for our models
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#hyperparameters
batch_size = 16
lr = 0.00002
epochs = 50

transform = transforms.Compose([
        transforms.ToTensor(),
        ])

dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

encoder = Encoder(16)
decoder = Decoder(16)

writer = SummaryWriter('/home/ujjawal/PycharmProjects/VAE/venv/log_dir/first_run')

optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr = lr, betas=(0.5, 0.999))

encoder.apply(weights_init)
decoder.apply(weights_init)


for epoch in range(epochs):
    reconstruct_loss = 0    #total reconstruction loss
    kl_loss = 0             #total kl divergence loss
    train_loss = 0          #total train loss(reconstruction + 2*kl loss)
    encoder.train()
    decoder.train()

    for i, (data, label) in enumerate(dataloader):
        prior = D.Normal(torch.zeros(16, ), torch.ones(16, 16))
        optimizer.zero_grad()
        encoded_op = encoder(data)          #output statistics for latent space
        z_mu = encoded_op[:, 0]
        z_logvar = encoded_op[:, 1]
        reconstruction_loss = 0            #loss for a batch
        epsilon = prior.sample()
        z = z_mu + epsilon * (z_logvar / 2).exp()
        output_data = decoder(z)
        reconstruction_loss += F.binary_cross_entropy(output_data, data.detach(), size_average=False)
        q = D.Normal(z_mu, (z_logvar / 2).exp())
        kld_loss = D.kl_divergence(q, prior).sum()
        reconstruct_loss += reconstruction_loss.item()
        kl_loss += kld_loss.item()
        loss = (reconstruction_loss + 2 * kld_loss)        #total loss for the processed batch
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if i % 500 == 0:
            print('[%d/%d][%d/%d]\tLoss: %.4f\t'
                  % (epoch, epochs, i, len(dataloader), loss))
        writer.add_scalars('Train loss', {'Reconstruction loss': reconstruction_loss / len(dataloader.dataset),
                                          'KL divergence': kld_loss / len(dataloader.dataset),
                                          'Train loss': loss / len(dataloader.dataset)}, epoch)

        sample = D.Normal(torch.zeros(16), torch.ones(16))         #sample to feed to generator to analyse its performance while training
        with torch.no_grad():
            output = decoder(sample.sample(torch.Size([64])))
        writer.add_image('Sample Image', torchvision.utils.make_grid(output, nrow=4, normalize=True), i)
        with torch.no_grad():
            output_reconstruct = decoder(z)
        writer.add_image('Reconstruction Image', torchvision.utils.make_grid(output_reconstruct, nrow=4, normalize=True), i)
writer.close()








