import torch
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D, Trainer1D, Dataset1D
import wandb
import pickle
from utils import set_all_seeds, preprocess_tensor

set_all_seeds(42)

device = 'cuda:2'
num_features = 600
training_steps = 15000
objective = 'pred_noise'
mode = 'train' # 'train' or 'reconstruct' or 'inpaint'
inpainting_strategy = 't-noised-replace' # 't-noised-replace' or 'original-replace' or 't-noised-replace'
dataset_name = 'dblp' # 'dota2' or 'dblp'

model = Unet1D(
    dim = 256,
    dim_mults = (1, 2, 4, 8),
    channels = 32
)

diffusion = GaussianDiffusion1D(
    model,
    seq_length = num_features,
    objective = objective,
    auto_normalize = False
).to(device)

if dataset_name == 'dota2':
    records_path = 'data/ae_t2v_dimSkill300_dimUser300_tFull_dota2.pkl'
    indices_path = 'data/dota2_train_test_indices.pkl'
elif dataset_name == 'dblp':
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

if dataset_name == 'dota2':
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

train_seq = preprocess_tensor(train_tensor)
val_seq = preprocess_tensor(val_tensor)
test_seq = preprocess_tensor(test_tensor)
all_seq = preprocess_tensor(all_tensor)
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


if mode == 'train':
    wandb.init(project='ddpm_pt_TF', name=f'ddpm_{dataset_name}_{training_steps}_{objective}')
    
    wandb.config.update({
        'training_steps': training_steps,
        'objective': objective,
        'mode': mode,
    })

    train_dataset = Dataset1D(train_seq)
    val_dataset = Dataset1D(val_seq)
    
    trainer = Trainer1D(
        diffusion,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_batch_size = 32,
        train_lr = 8e-5,
        train_num_steps = training_steps, # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        eval_every = 50,    # save model and sample every n steps
    )
    
    best_model, final_model = trainer.train()
    
    torch.save(best_model.model.state_dict(), f'checkpoints/ddpm_{dataset_name}_{training_steps}_{objective}_best.pt')
    torch.save(final_model.model.state_dict(), f'checkpoints/ddpm_{dataset_name}_{training_steps}_{objective}_final.pt')
    
    wandb.finish()
    
elif mode == 'reconstruct':
    # load the trained model
    diffusion.model.load_state_dict(torch.load(f'checkpoints/ddpm_{dataset_name}_{training_steps}_{objective}_best.pt'))
    
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

    with open(f'output/reconstructed_{dataset_name}_model-steps-{training_steps}_obj-{objective}.pkl', 'wb') as f:
        pickle.dump(denoised_records, f)

elif mode == 'inpaint':
    # load the trained model
    diffusion.model.load_state_dict(torch.load(f'checkpoints/ddpm_{dataset_name}_{training_steps}_{objective}_best.pt'))
    
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
    denoised_all_seq = diffusion.inpaint(noisy_all_seq, all_seq, noise, strategy=inpainting_strategy)
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

    with open(f'output/inpainted_{dataset_name}_model-steps-{training_steps}_obj-{objective}.pkl', 'wb') as f:
        pickle.dump(denoised_records, f)
