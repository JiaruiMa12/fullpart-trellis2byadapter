import importlib

__attributes = {
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    'SparseStructureFlowModel': 'sparse_structure_flow',
    'SLatEncoder': 'structured_latent_vae',
    'SLatGaussianDecoder': 'structured_latent_vae',
    'SLatRadianceFieldDecoder': 'structured_latent_vae',
    'SLatMeshDecoder': 'structured_latent_vae',
    'SLatFlowModel': 'structured_latent_flow',
    'FlexiDualGridVaeEncoder': 'structured_latent_vae',
    'FlexiDualGridVaeDecoder': 'structured_latent_vae',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    from safetensors.torch import load_file
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Get the model class to inspect its __init__ signature
    model_class = __getattr__(config['name'])
    import inspect
    model_params = inspect.signature(model_class.__init__).parameters
    
    # Filter out unsupported arguments from config['args']
    filtered_args = {k: v for k, v in config['args'].items() if k in model_params}
    
    # Special handling for FlexiDualGridVae models - convert list-based args to integer
    # The current TRELLIS submodule doesn't have the actual FlexiDualGridVae implementation,
    # only aliases to SLatEncoder/SLatMeshDecoder which expect integer model_channels
    if config['name'] in ['FlexiDualGridVaeEncoder', 'FlexiDualGridVaeDecoder']:
        if 'model_channels' in filtered_args and isinstance(filtered_args['model_channels'], list):
            # Use the first (largest) channel dimension
            filtered_args['model_channels'] = filtered_args['model_channels'][0]
        if 'num_blocks' in filtered_args and isinstance(filtered_args['num_blocks'], list):
            # Sum the blocks for total number
            filtered_args['num_blocks'] = sum(filtered_args['num_blocks'])
        # Override resolution to avoid CUDA OOM (original 256 is too large for mesh extractor)
        if 'resolution' in filtered_args:
            filtered_args['resolution'] = 64  # Reduce from 256 to 64
        # Add default representation_config if not present
        if 'representation_config' not in filtered_args or filtered_args['representation_config'] is None:
            filtered_args['representation_config'] = {'use_color': False}
    
    # Also filter kwargs
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in model_params}
    
    model = model_class(**filtered_args, **filtered_kwargs)
    # Load with strict=False to handle architecture mismatches (e.g., resolution override)
    model.load_state_dict(load_file(model_file), strict=False)

    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_vae import SLatEncoder, SLatGaussianDecoder, SLatRadianceFieldDecoder, SLatMeshDecoder
    from .structured_latent_flow import SLatFlowModel
