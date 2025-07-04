import torch
# ... other imports

def load_checkpoint(model, weights, map_location=None): # <--- ADDED map_location=None
    """Loads a checkpoint into the model."""
    print(f"Loading weights from {weights}...")
    checkpoint = torch.load(weights, map_location=map_location) # <--- PASSED map_location here
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except:
        # If the model was saved with DataParallel, state_dict keys might have 'module.' prefix
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v # remove 'module.' prefix
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    print("Checkpoint loaded successfully.")
    # You might also want to modify load_optim if you use it and it loads optimizer states
    # that were saved on GPU. It would need a similar map_location argument.