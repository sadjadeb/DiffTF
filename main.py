import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import wandb
import pickle
from utils import set_all_seeds, preprocess_tensor
import argparse

set_all_seeds(42)


parser = argparse.ArgumentParser(description='Diffusion Model Training and Inference')
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for computation')
parser.add_argument('--dataset_name', type=str, default='dblp', help='Name of the dataset', choices=['dota2', 'dblp'])
parser.add_argument('--task', type=str, default='train', help='Task to perform: train, reconstruct, inpaint', choices=['train', 'reconstruct', 'inpaint'])
parser.add_argument('--training_steps', type=int, default=45000, help='Number of training steps')
parser.add_argument('--train_objective', type=str, default='pred_noise', help='Objective for training the diffusion model', choices=['pred_noise', 'pred_x0'])
parser.add_argument('--model_hidden_dim', type=int, default=256, help='Dimension of the model')
parser.add_argument('--inpainting_strategy', type=str, default='t-noised-replace', help='Inpainting strategy', choices=['t-noised-replace', 'original-replace', 't-noised-replace'])
parser.add_argument('--trained_model', type=str, default=None, help='Path to a pre-trained model for inference')

args = parser.parse_args()

if args.task == 'inpaint' or args.task == 'reconstruct':
    assert args.trained_model is not None, "Please provide the path to the trained model"

device = args.device

model = Unet1D(
    dim = args.model_hidden_dim,
    dim_mults = (1, 2, 4, 8),
    channels = 32 if args.dataset_name == 'dblp' else 2,
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = 600, # Being hardcoded as the output of the T2V model is 600
    objective = args.train_objective,
    auto_normalize = False
).to(device)

if args.dataset_name == 'dota2':
    records_path = 'data/ae_t2v_dimSkill300_dimUser300_tFull_dota2.pkl'
    indices_path = 'data/dota2_train_test_indices.pkl'
elif args.dataset_name == 'dblp':
    records_path = 'data/ae_t2v_dimSkill300_dimUser300_tFull_dataset_V2.2.pkl'
    indices_path = 'data/Train_Test_indices_V2.3.pkl'
else:
    raise ValueError('Invalid dataset')


with open(records_path, 'rb') as f:
    records = pickle.load(f)
    
with open(indices_path, 'rb') as f:
    indices = pickle.load(f)

# Convert records into a dictionary for easy access
records_dict = {rec[0]: torch.cat([torch.tensor(rec[1]), torch.tensor(rec[2])]) for rec in records}

# Get the first fold
fold = indices[1]
train_ids = fold["Train"]
test_ids = fold["Test"]

if args.dataset_name == 'dota2':
    # limit the indices in each fold to to 141
    train_ids = [i for i in train_ids if i < 141]
    test_ids = [i for i in test_ids if i < 141]

# Construct tensors for train and test sets
train_val_tensor = torch.stack([records_dict[i] for i in train_ids])
test_tensor = torch.stack([records_dict[i] for i in test_ids])
all_tensor = torch.stack([records_dict[i] for i in train_ids + test_ids])

# Define the split ratio (e.g., 80% train, 20% validation)
train_size = int(0.8 * len(train_val_tensor))  # 80% of the dataset
train_tensor = train_val_tensor[:train_size]
val_tensor = train_val_tensor[train_size:]

print("Train Tensor Shape:", train_tensor.shape)
print("Validation Tensor Shape:", val_tensor.shape)
print("Test Tensor Shape:", test_tensor.shape)
print("All Tensor Shape:", all_tensor.shape)

num_target_channels = 32 if args.dataset_name == 'dblp' else 2

train_seq = preprocess_tensor(train_tensor, target_channels=num_target_channels)
val_seq = preprocess_tensor(val_tensor, target_channels=num_target_channels)
test_seq = preprocess_tensor(test_tensor, target_channels=num_target_channels)
all_seq = preprocess_tensor(all_tensor, target_channels=num_target_channels)
print("Train Seq Shape:", train_seq.shape)
print("Validation Seq Shape:", val_seq.shape)
print("Test Seq Shape:", test_seq.shape)
print("All Seq Shape:", all_seq.shape)


def reorder_list(reference_list, target_list):
    # Create a dictionary from the target list for quick lookup
    target_dict = {item[0]: item for item in target_list}
    
    # Reorder the target list based on the reference list's order
    ordered_target_list = [target_dict[item[0]] for item in reference_list if item[0] in target_dict]
    
    return ordered_target_list


if args.task == 'train':
    wandb.init(project='ddpm_pt_TF', name=f'ddpm_{args.dataset_name}')
    
    wandb.config.update({
        'training_steps': args.training_steps,
        'train_objective': args.train_objective,
        'model_hidden_dim': args.model_hidden_dim,
        'dataset_name': args.dataset_name
    })

    train_dataset = Dataset1D(train_seq)
    val_dataset = Dataset1D(val_seq)
    
    trainer = Trainer1D(
        diffusion,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_batch_size = 32,
        train_lr = 8e-5,
        train_num_steps = args.training_steps, # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        eval_every = 50,    # save model and sample every n steps
    )
    
    best_model, final_model = trainer.train()
    
    torch.save(best_model.model.state_dict(), f'checkpoints/ddpm_{args.dataset_name}_{args.training_steps}_{args.train_objective}_{args.model_hidden_dim}_best.pt')
    torch.save(final_model.model.state_dict(), f'checkpoints/ddpm_{args.dataset_name}_{args.training_steps}_{args.train_objective}_{args.model_hidden_dim}_final.pt')
    
    wandb.finish()
    
elif args.task == 'reconstruct':
    # load the trained model
    diffusion.model.load_state_dict(torch.load(args.trained_model))
    
    # add noise to the sequence
    b = all_seq.shape[0]
    t = torch.full((b,), diffusion.num_timesteps - 1, device=device)
    noise = torch.randn_like(all_seq, device=device)
    noisy_all_seq = diffusion.q_sample(x_start=all_seq.to(device), t=t, noise=noise)
    
    denoised_all_seq = diffusion.denoise(noisy_all_seq)
    denoised_all_seq = denoised_all_seq.detach().cpu()
    
    # calculate the MSE between the denoised sequence and the original sequence
    mse = torch.nn.functional.mse_loss(denoised_all_seq, all_seq).item()
    print(f'MSE: {mse}')
    
    # reshape the denoised sequence to (n_samples, 600)
    denoised_all_seq = denoised_all_seq.numpy().reshape(-1, 600)
    
    # Save the denoised sequence like the input pickle file
    denoised_records = []
    for i, original_index in enumerate((train_ids + test_ids)[:len(denoised_all_seq)]):
        denoised_records.append((original_index, denoised_all_seq[i][:300], denoised_all_seq[i][300:]))
        
    print("Denoised Records Length:", len(denoised_records))

    with open(f'output/reconstructed_{args.trained_model.split("/")[-1].replace(".pt", "")}.pkl', 'wb') as f:
        pickle.dump(denoised_records, f)

elif args.task == 'inpaint':
    # load the trained model
    diffusion.model.load_state_dict(torch.load(args.trained_model))
    
    # replace the second half of the sequence with zeros
    masked_all_seq = all_seq.clone()
    masked_all_seq[:, :, 300:] = torch.randn_like(masked_all_seq[:, :, 300:])
    
    # add noise to the sequence
    b = masked_all_seq.shape[0]
    t = torch.full((b,), diffusion.num_timesteps - 1, device=device)
    noise = torch.randn_like(masked_all_seq, device=device)
    masked_all_seq = masked_all_seq.to(device)
    noisy_all_seq = diffusion.q_sample(x_start=masked_all_seq, t=t, noise=noise)
    
    # reconstruct the masked sequence
    all_seq = all_seq.to(device)
    denoised_all_seq = diffusion.inpaint(noisy_all_seq, all_seq, noise, strategy=args.inpainting_strategy)
    denoised_all_seq = denoised_all_seq.detach().cpu()
    all_seq = all_seq.detach().cpu()
    
    # calculate the MSE between the denoised sequence and the original sequence
    mse = torch.nn.functional.mse_loss(denoised_all_seq, all_seq).item()
    print(f'MSE: {mse}')
    
    # reshape the denoised sequence to (n_samples, 600)
    denoised_all_seq = denoised_all_seq.numpy().reshape(-1, 600)
    
    # Save the denoised sequence like the input pickle file
    denoised_records = []
    for i, original_index in enumerate((train_ids + test_ids)):
        try:
            denoised_records.append((original_index, denoised_all_seq[i][:300], denoised_all_seq[i][300:]))
        except:
            # as we truncated the last records to fit the channel size, we may have less records than the original
            # in this case, we put full zeros for the missing records
            denoised_records.append((original_index, [0]*300, [0]*300))
        
    # sort the records based on the original order
    denoised_records = reorder_list(records, denoised_records)
    
    print("Denoised Records Length:", len(denoised_records))

    with open(f'output/inpainted_{args.trained_model.split("/")[-1].replace(".pt", "")}.pkl', 'wb') as f:
        pickle.dump(denoised_records, f)
