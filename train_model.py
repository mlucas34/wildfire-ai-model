import torch
from torch import nn, optim
import numpy as np
from model import Encoder, Decoder, WildfireSeq2Seq
from WildFire_partially_observable import WildFireEnv
import os
from dataset import WildFireDataset
from torch.utils.data import DataLoader
import json

n_grid = 5
input_dim = 48
output_dim = 16
encoder_embedding_dim = 256
decoder_embedding_dim = 256
hidden_dim = 512
n_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 500
batch_size = 5

def train(model, data_loader, optimizer, loss_function, clip, teacher_forcing_ratio, device):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (src, trg) in enumerate(data_loader):
            src = src.to(device)  # (seq_len, batch, feat)
            trg = trg.to(device)  # (batch, seq_len, feat)

            optimizer.zero_grad()

            # model returns (batch, seq_len, feat)
            outputs = model(src, trg, teacher_forcing_ratio)
            outputs = torch.log_softmax(outputs, dim=-1)

            if outputs.dtype != torch.float32:
                outputs = outputs.float()

            loss = loss_function(outputs.transpose(1, 2), trg)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()

            if batch_idx == 0 and (epoch + 1) % 5 == 0:
                with torch.no_grad():
                    probs = torch.exp(outputs) # convert to probabilities

                    predicted_state = torch.argmax(probs, dim=-1)  # (batch, seq_len)
                    actual_state = trg.argmax(dim=-1) if trg.ndim == 3 else trg

                    print(f"\nEpoch {epoch + 1}, Batch {batch_idx+1}")
                    print("Predicted state (first sample):")
                    print(predicted_state[:5].cpu().numpy())
                    print("Actual state (first sample):")
                    print(actual_state[:5].cpu().numpy())
                    print()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(data_loader):.6f}")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()
            }, model_path)



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    enc = Encoder(input_dim, encoder_embedding_dim,
                  hidden_dim, n_layers, encoder_dropout)
    dec = Decoder(output_dim, decoder_embedding_dim,
                  hidden_dim, n_layers, decoder_dropout)

    model = WildfireSeq2Seq(enc, dec, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)#change this 
    loss_function = nn.NLLLoss()

    model_path = "best_model.pth"
    samples_json_path = "samples.json"

    if os.path.exists(model_path):
        print(f"loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    env = WildFireEnv(method="hypRL", n_grid=n_grid)
    env.init_agents()
    env.reset()
    sample_obs = env.get_agent_obs(0)

    dataset = WildFireDataset(env, num_episodes = 300, steps = 20, agent_idx = 0)
    dataset.collect_data()

    serializable = [{"src": s.tolist(), "trg": t.tolist()} for s,t in dataset.samples]

    with open(samples_json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": {"n": len(serializable)}, "data": serializable}, f)


    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    print("starting training")
    train(model, train_loader, optimizer, loss_function, None, 0.5, device)
    
    # save progress
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)
    print(f"model saved to {model_path}")

