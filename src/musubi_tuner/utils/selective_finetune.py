"""
Selective Fine-tuning Utilities for Ultra-Low VRAM Training

This module implements memory-efficient selective fine-tuning by training only a small, 
deterministic subset of parameters while keeping the main model in CPU memory.

Key features:
- Deterministic parameter selection based on --ff (fraction) and --ffid (ID)
- Even distribution across all model components (embeddings, blocks, head)
- Proxy parameter system for ultra-low VRAM usage
- Full compatibility with existing training infrastructure
"""

import logging
import random
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict

# Production flag to disable all selective fine-tuning debug messages
TURN_ON_DEBUGS = False

logger = logging.getLogger(__name__)


def debug_log(level, message, *args, **kwargs):
    """Conditional logging based on TURN_ON_DEBUGS flag"""
    if TURN_ON_DEBUGS:
        getattr(logger, level)(message, *args, **kwargs)


# Override logger methods to be conditional for selective fine-tuning
class ConditionalLogger:
    def __init__(self, original_logger):
        self._logger = original_logger
    
    def info(self, message, *args, **kwargs):
        if TURN_ON_DEBUGS:
            self._logger.info(message, *args, **kwargs)
    
    def debug(self, message, *args, **kwargs):
        if TURN_ON_DEBUGS:
            self._logger.debug(message, *args, **kwargs)
    
    def warning(self, message, *args, **kwargs):
        if TURN_ON_DEBUGS:
            self._logger.warning(message, *args, **kwargs)
    
    def error(self, message, *args, **kwargs):
        # Always log errors regardless of debug flag
        self._logger.error(message, *args, **kwargs)

# Replace logger with conditional version
logger = ConditionalLogger(logger)


def get_all_trainable_parameters(model: nn.Module) -> Dict[str, nn.Parameter]:
    """
    Extract all trainable parameters from the model with their names.
    
    Args:
        model: The PyTorch model to extract parameters from
        
    Returns:
        Dictionary mapping parameter names to parameter tensors
    """
    params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            params[name] = param
    return params


def select_parameters_deterministic(
    all_params: Dict[str, nn.Parameter],
    fraction: float,
    selection_id: int,
    return_slice_metadata: bool = False
) -> Union[List[str], Tuple[List[str], Dict]]:
    """
    Selects a deterministic, non-overlapping, and evenly distributed fraction of model parameters.
    """
    
    original_state = random.getstate()
    random.seed(42)

    # Deterministic shuffle for the local tie-breaker
    attn_components = ['q', 'k', 'v', 'o']
    random.shuffle(attn_components)
    attn_component_priority = {name: i for i, name in enumerate(attn_components)}

    def get_param_priority(param_name: str) -> int:
        name_parts = param_name.split('.')
        # This part is for stripping block numbers to generalize the component name
        if len(name_parts) > 2 and name_parts[0] == 'blocks' and name_parts[1].isdigit():
            param_name = '.'.join(name_parts[2:])
        # This is the simple priority scheme
        if ('attn' in param_name or 'ffn' in param_name) and 'weight' in param_name: return 0
        if ('attn' in param_name or 'ffn' in param_name) and 'bias' in param_name: return 1
        if 'norm' in param_name: return 2
        return 3

    # --- Helper function for the new, more detailed LOCAL sorting within blocks ---
    def get_local_sort_key(param_info: Tuple[str, int]) -> tuple:
        name, count = param_info
        
        # Default sort is by size, then name
        key = (count, name)
        
        # For attn weights of the same size, we introduce a deterministic sub-priority
        local_name = name
        name_parts = name.split('.')
        if len(name_parts) > 2 and name_parts[0] == 'blocks' and name_parts[1].isdigit():
            local_name = '.'.join(name_parts[2:])
            
        if 'attn' in local_name and 'weight' in local_name:
            component = local_name.split('.')[-2]
            if component in attn_component_priority:
                # Tie-break by component priority, THEN by name
                key = (count, attn_component_priority[component], name)
                
        return key

    try:
        total_model_params = sum(p.numel() for p in all_params.values())
        target_param_count = max(1, int(total_model_params * fraction))
        logger.info(f"Target: {target_param_count:,} params ({fraction:.6f}) for set {selection_id}")

        all_param_list = sorted([(n, p.numel()) for n, p in all_params.items()],
                                key=lambda x: (get_param_priority(x[0]), x[1], x[0]))

        num_sets = 8
        partitioned_pool = [
            (n, c) for idx, (n, c) in enumerate(all_param_list)
            if idx % num_sets == selection_id
        ]
        logger.info(f"Partitioned pool size: {len(partitioned_pool)} tensors")
        
        block_params = defaultdict(list)
        non_block_params = []
        for name, count in partitioned_pool:
            if 'blocks.' in name and name.split('.')[1].isdigit():
                block_num = int(name.split('.')[1])
                block_params[block_num].append((name, count))
            else:
                non_block_params.append((name, count))

        candidate_pool = []
        

        for block_num in sorted(block_params.keys()):
            block_params[block_num].sort(key=get_local_sort_key)
        
        max_depth = max(len(p) for p in block_params.values()) if block_params else 0
        for rank in range(max_depth):
            for block_num in sorted(block_params.keys()):
                if rank < len(block_params[block_num]):
                    candidate_pool.append(block_params[block_num][rank])
        
        non_block_params.sort(key=lambda x: (x[1], x[0]))
        candidate_pool.extend(non_block_params)
        
        logger.info(f"Created a balanced candidate pool of {len(candidate_pool)} tensors.")
        
        selected_params = {}
        current_sum = 0
        for name, count in candidate_pool:
            selected_params[name] = count
            current_sum += count
            
            if current_sum >= target_param_count:
                break

        selected_param_names = list(selected_params.keys())
        logger.info(f"Final selection: {len(selected_params)} tensors, {current_sum:,} params")
        
        if return_slice_metadata:
            slice_metadata = {}
            component_distribution = defaultdict(int)
            block_distribution = defaultdict(int)

            for name, count in selected_params.items():
                if 'blocks.' in name:
                    component_distribution['blocks'] += count
                    try:
                        block_num = int(name.split('.')[1])
                        block_distribution[f'block_{block_num}'] += count
                    except (ValueError, IndexError):
                        component_distribution['other'] += count
                elif 'embedding' in name:
                    component_distribution['embeddings'] += count
                elif 'head' in name:
                    component_distribution['head'] += count
                else:
                    component_distribution['other'] += count

            slice_metadata['total_model_params'] = total_model_params
            slice_metadata['target_param_count'] = target_param_count
            slice_metadata['selected_param_count'] = current_sum
            slice_metadata['selected_tensor_count'] = len(selected_param_names)
            slice_metadata['actual_fraction'] = current_sum / total_model_params if total_model_params > 0 else 0
            slice_metadata['component_distribution'] = dict(component_distribution)
            slice_metadata['block_distribution'] = dict(block_distribution)
            
            return selected_param_names, slice_metadata
        else:
            return selected_param_names

    finally:
        random.setstate(original_state)


def create_proxy_parameters(
    model: nn.Module, 
    selected_param_names: List[str],
    slice_metadata: Dict[str, Tuple[int, int]] = None
) -> Tuple[nn.ParameterDict, Dict[str, Tuple[str, torch.Size, torch.dtype]]]:
    """
    Create proxy parameters for the selected subset and replace original parameters.
    
    Args:
        model: The model containing the original parameters
        selected_param_names: Names of parameters to create proxies for
        slice_metadata: Optional dict mapping param names to (slice_count, original_count)
        
    Returns:
        Tuple of (proxy_parameters, parameter_metadata)
        - proxy_parameters: ParameterDict containing proxy parameters
        - parameter_metadata: Metadata for mapping back to original parameters
    """
    proxy_params = nn.ParameterDict()
    param_metadata = {}
    slice_metadata = slice_metadata or {}
    
    model_params = dict(model.named_parameters())
    
    for i, param_name in enumerate(selected_param_names):
        if param_name not in model_params:
            logger.warning(f"Parameter {param_name} not found in model")
            continue
            
        original_param = model_params[param_name]
        
        # Check if this parameter needs slicing
        if param_name in slice_metadata:
            slice_count, original_count = slice_metadata[param_name]
            # Calculate slice dimensions - slice from the first dimension
            original_shape = original_param.shape
            original_numel = original_param.numel()
            
            if original_numel != original_count:
                logger.warning(f"Mismatch: {param_name} has {original_numel} elements, expected {original_count}")
            
            # Calculate how to slice the first dimension to get approximately slice_count parameters
            first_dim = original_shape[0]
            slice_first_dim = max(1, int((slice_count / original_numel) * first_dim))
            slice_shape = (slice_first_dim,) + original_shape[1:]
            
            # Create proxy parameter with sliced shape
            proxy_name = f"proxy_{i:06d}"
            proxy_param = nn.Parameter(
                torch.zeros(slice_shape, dtype=original_param.dtype, device='cpu'),
                requires_grad=True
            )
            
            # Copy sliced values from original parameter
            with torch.no_grad():
                proxy_param.data.copy_(original_param.data[:slice_first_dim])
            
            logger.info(f"Sliced {param_name}: {original_shape} -> {slice_shape} ({proxy_param.numel():,} params)")
        else:
            # Create proxy parameter with same shape and dtype (no slicing)
            proxy_name = f"proxy_{i:06d}"
            proxy_param = nn.Parameter(
                torch.zeros_like(original_param, device='cpu'),
                requires_grad=True
            )
            
            # Copy current values from original parameter
            with torch.no_grad():
                proxy_param.data.copy_(original_param.data)
        
        proxy_params[proxy_name] = proxy_param
        param_metadata[proxy_name] = (param_name, original_param.shape, original_param.dtype)
        
        # CRITICAL: Replace the original parameter in the model with the proxy parameter
        # This ensures gradients flow to the proxy parameter during training
        # Note: Keep proxy on CPU initially, will be moved to correct device later
        _replace_model_parameter(model, param_name, proxy_param)
        
        # CRITICAL FIX: Verify the parameter was actually replaced and is accessible for gradients
        verification_params = dict(model.named_parameters())
        if param_name in verification_params:
            replaced_param = verification_params[param_name]
            if id(replaced_param) == id(proxy_param):
                logger.debug(f"✓ Parameter replacement verified for {param_name}")
                # Ensure the replaced parameter maintains gradient computation
                if not replaced_param.requires_grad:
                    logger.warning(f"Fixed requires_grad for replaced parameter {param_name}")
                    replaced_param.requires_grad = True
            else:
                logger.error(f"✗ Parameter replacement failed for {param_name} - IDs don't match")
        else:
            logger.error(f"✗ Parameter {param_name} not found after replacement")
        
        logger.debug(f"Created and replaced proxy {proxy_name} for {param_name} "
                    f"shape={original_param.shape}")
    
    total_proxy_params = sum(p.numel() for p in proxy_params.values())
    logger.info(f"Created {len(proxy_params)} proxy parameters "
               f"with {total_proxy_params:,} total elements")
    
    return proxy_params, param_metadata


def _replace_model_parameter(model: nn.Module, param_name: str, new_param: nn.Parameter):
    """
    Replace a parameter in the model with a new parameter.
    
    Args:
        model: The model to modify
        param_name: Dot-separated path to the parameter (e.g., "blocks.0.self_attn.q.weight")
        new_param: The new parameter to use
    """
    # Navigate to the parameter in the model
    module = model
    param_path = param_name.split('.')
    param_attr_name = param_path[-1]
    
    # Navigate to the parent module
    for part in param_path[:-1]:
        if hasattr(module, part):
            module = getattr(module, part)
        else:
            # Handle indexed access like "blocks.0" -> module.blocks[0]
            try:
                if '.' in part:
                    # This shouldn't happen with our current split, but just in case
                    subparts = part.split('.')
                    for subpart in subparts:
                        module = getattr(module, subpart)
                else:
                    # Try direct attribute access first
                    try:
                        module = getattr(module, part)
                    except AttributeError:
                        # If that fails and it's a digit, try indexed access on previous module
                        if part.isdigit() and hasattr(module, '__getitem__'):
                            module = module[int(part)]
                        else:
                            raise
            except (AttributeError, ValueError, IndexError, TypeError) as e:
                logger.error(f"Could not navigate to parameter {param_name}: {e}")
                return
    
    # Replace the parameter
    if hasattr(module, param_attr_name):
        old_param = getattr(module, param_attr_name)
        
        # PHASE 1 DEBUG: Log parameter states before replacement
        logger.debug(f"[REPLACE DEBUG] {param_name}: old_requires_grad={old_param.requires_grad}, new_requires_grad={new_param.requires_grad if isinstance(new_param, nn.Parameter) else 'N/A'}")
        
        # CRITICAL FIX: Ensure new parameter has requires_grad=True before replacing
        if isinstance(new_param, nn.Parameter) and not new_param.requires_grad:
            logger.warning(f"Fixing requires_grad=False for {param_name} before replacement")
            new_param.requires_grad = True
            
        setattr(module, param_attr_name, new_param)
        logger.info(f"Successfully replaced parameter {param_name} in model (was {type(old_param)}, now {type(new_param)})")
        
        # Verify the replacement worked and that requires_grad is correct
        verification_param = getattr(module, param_attr_name)
        if id(verification_param) == id(new_param):
            logger.debug(f"✓ Parameter replacement verified for {param_name}, requires_grad={verification_param.requires_grad}")
            
            # Double-check requires_grad
            if not verification_param.requires_grad:
                logger.warning(f"⚠ Parameter {param_name} has requires_grad=False after replacement, fixing...")
                verification_param.requires_grad = True
                logger.debug(f"✓ Fixed requires_grad for {param_name}, now={verification_param.requires_grad}")
        else:
            logger.error(f"✗ Parameter replacement failed for {param_name} - objects don't match")
    else:
        logger.error(f"Parameter {param_attr_name} not found in module for {param_name}")


def apply_proxy_updates_to_model(
    model: nn.Module,
    proxy_params: nn.ParameterDict,
    param_metadata: Dict[str, Tuple[str, torch.Size, torch.dtype]]
) -> None:
    """
    Apply the trained proxy parameter updates back to the original model.
    
    Args:
        model: The model to update
        proxy_params: The trained proxy parameters
        param_metadata: Metadata mapping proxy names to original parameter info
    """
    model_params = dict(model.named_parameters())
    updated_count = 0
    
    for proxy_name, proxy_param in proxy_params.items():
        if proxy_name not in param_metadata:
            logger.warning(f"No metadata found for proxy parameter {proxy_name}")
            continue
            
        original_name, expected_shape, expected_dtype = param_metadata[proxy_name]
        
        if original_name not in model_params:
            logger.warning(f"Original parameter {original_name} not found in model")
            continue
            
        original_param = model_params[original_name]
        
        # Validate shapes match
        if proxy_param.shape != expected_shape:
            logger.error(f"Shape mismatch for {original_name}: "
                        f"proxy={proxy_param.shape}, expected={expected_shape}")
            continue
            
        # Update the original parameter with proxy values
        with torch.no_grad():
            # Ensure data is on the same device and dtype as original
            updated_data = proxy_param.data.to(
                device=original_param.device,
                dtype=original_param.dtype
            )
            original_param.data.copy_(updated_data)
            
        updated_count += 1
        logger.debug(f"Updated {original_name} from proxy {proxy_name}")
    
    logger.info(f"Applied updates from {updated_count} proxy parameters to original model")


class SelectiveFinetuneManager:
    """
    Main class for managing selective fine-tuning with ultra-low VRAM usage.
    
    This class handles:
    - Parameter selection and proxy creation
    - Training setup with proxy parameters
    - Applying updates back to the full model
    """
    
    def __init__(self, model: nn.Module, fraction: float, selection_id: int):
        """
        Initialize the selective fine-tuning manager.
        
        Args:
            model: The DiT model to selectively fine-tune
            fraction: Fraction of parameters to train (e.g., 0.00001)
            selection_id: ID for deterministic parameter selection
        """
        self.model = model
        self.fraction = fraction
        self.selection_id = selection_id
        self.proxy_params = None
        self.param_metadata = None
        self.selected_param_names = None
        
        logger.info(f"Initializing SelectiveFinetuneManager with "
                   f"fraction={fraction:.6f}, selection_id={selection_id}")
        
        # PHASE 1 DEBUG: Check initial model parameter states
        logger.info("=== PHASE 1 DEBUG: Initial Model Parameter States ===")
        total_params = 0
        grad_enabled_params = 0
        for name, param in model.named_parameters():
            total_params += 1
            if param.requires_grad:
                grad_enabled_params += 1
        logger.info(f"Model has {total_params} total parameters, {grad_enabled_params} with requires_grad=True")
        
        # Initialize the selective fine-tuning setup
        self._setup_selective_training()
    
    def _setup_selective_training(self):
        """Setup proxy parameters for selective training."""
        # Get all trainable parameters
        all_params = get_all_trainable_parameters(self.model)
        
        # Select subset deterministically (request slice metadata for slicing support)
        self.selected_param_names, self.slice_metadata = select_parameters_deterministic(
            all_params, self.fraction, self.selection_id, return_slice_metadata=True
        )
        
        # Create proxy parameters (with slicing support)
        self.proxy_params, self.param_metadata = create_proxy_parameters(
            self.model, self.selected_param_names, self.slice_metadata
        )
        
        # PHASE 2 FIX: Ensure all parameters have requires_grad=True after creation
        self._ensure_proxy_params_require_grad("after_parameter_creation")
        
        logger.info("Selective fine-tuning setup complete")
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """
        Get the proxy parameters that are now in the model and should be trained.
        
        Returns:
            List of proxy parameters for the optimizer (now in the model)
        """
        if self.proxy_params is None or self.param_metadata is None:
            raise RuntimeError("Selective fine-tuning not initialized")
        
        # Return the parameters that are now actually in the model
        model_params = dict(self.model.named_parameters())
        trainable_params = []
        requires_grad_fixes = 0
        identity_mismatches = 0
        
        for proxy_name in self.param_metadata:
            original_name, _, _ = self.param_metadata[proxy_name]
            if original_name in model_params:
                param = model_params[original_name]
                
                # CRITICAL: Verify this is actually our proxy parameter
                if proxy_name in self.proxy_params:
                    proxy_param = self.proxy_params[proxy_name]
                    if id(param) != id(proxy_param):
                        identity_mismatches += 1
                        logger.error(f"Parameter identity mismatch for {original_name}: model_id={id(param)}, proxy_id={id(proxy_param)}")
                
                # CRITICAL FIX: Ensure the parameter has requires_grad=True
                if not param.requires_grad:
                    logger.warning(f"Parameter {original_name} had requires_grad=False, fixing to True")
                    param.requires_grad = True
                    requires_grad_fixes += 1
                
                trainable_params.append(param)
        
        # Log critical issues
        if identity_mismatches > 0:
            logger.error(f"=== CRITICAL: {identity_mismatches} parameter identity mismatches detected ===")
            logger.error("This will prevent training from working correctly!")
        
        if requires_grad_fixes > 0:
            logger.warning(f"Fixed requires_grad=False for {requires_grad_fixes}/{len(trainable_params)} parameters")
        
        logger.debug(f"Returning {len(trainable_params)} trainable parameters from model")
        return trainable_params
    
    def _ensure_proxy_params_require_grad(self, context: str = "unknown"):
        """
        PHASE 2 FIX: Ensure all proxy parameters maintain requires_grad=True.
        This should be called at key points in the setup process.
        
        Args:
            context: Description of when this is called for debugging
        """
        if self.proxy_params is None or self.param_metadata is None:
            return
            
        logger.debug(f"=== PHASE 2 FIX: Ensuring requires_grad=True ({context}) ===")
        fixes_applied = 0
        
        # Fix requires_grad in our proxy_params dict
        for proxy_name, proxy_param in self.proxy_params.items():
            if not proxy_param.requires_grad:
                logger.warning(f"[{context}] Fixing requires_grad=False for proxy {proxy_name}")
                proxy_param.requires_grad = True
                fixes_applied += 1
        
        # Fix requires_grad in the model parameters
        model_params = dict(self.model.named_parameters())
        for proxy_name in self.param_metadata:
            original_name, _, _ = self.param_metadata[proxy_name]
            if original_name in model_params:
                param = model_params[original_name]
                if not param.requires_grad:
                    logger.warning(f"[{context}] Fixing requires_grad=False for model param {original_name}")
                    param.requires_grad = True
                    fixes_applied += 1
        
        if fixes_applied > 0:
            logger.warning(f"=== PHASE 2 FIX: Applied {fixes_applied} requires_grad fixes in context '{context}' ===")
        else:
            logger.debug(f"All proxy parameters already have requires_grad=True in context '{context}'")
    
    def move_proxy_params_to_device(self, device: torch.device):
        """
        Move proxy parameters to the specified device.
        CRITICAL FIX: Preserve parameter object identity to maintain optimizer connections.
        
        Args:
            device: Device to move proxy parameters to
        """
        if self.proxy_params is None:
            raise RuntimeError("Selective fine-tuning not initialized")
        
        logger.info(f"=== FINAL FIX: Moving proxy parameters to {device} (preserving object identity) ===")
        
        # Move proxy parameters to device using in-place operations to preserve identity
        for proxy_name, proxy_param in self.proxy_params.items():
            if proxy_name in self.param_metadata:
                original_name, _, original_dtype = self.param_metadata[proxy_name]
                
                # FINAL FIX: Use in-place operations to preserve parameter object identity
                logger.debug(f"[IDENTITY FIX] {proxy_name}: before_id={id(proxy_param)}, device={proxy_param.device}")
                
                # Move data in-place (this preserves the Parameter object identity)
                proxy_param.data = proxy_param.data.to(device=device, dtype=original_dtype)
                proxy_param.requires_grad = True  # Ensure gradient computation
                
                # Verify the Parameter object ID didn't change
                logger.debug(f"[IDENTITY FIX] {proxy_name}: after_id={id(proxy_param)}, device={proxy_param.device}")
                
                # Update in model - but the Parameter object is the same!
                _replace_model_parameter(self.model, original_name, proxy_param)
                
        logger.info(f"Moved proxy parameters to {device} (object identities preserved)")
        
        # PHASE 2 FIX: Ensure all parameters still have requires_grad=True after device movement
        self._ensure_proxy_params_require_grad("after_device_movement")
        
        # FINAL FIX: Verify parameter IDs match between our dict and the model
        logger.info("=== FINAL FIX: Verifying parameter identity consistency ===")
        trainable_params = self.get_trainable_parameters()
        grad_count = sum(1 for p in trainable_params if p.requires_grad)
        logger.info(f"After device movement: {grad_count}/{len(trainable_params)} model parameters have requires_grad=True")
        
        # Check that parameter objects are identical between proxy_params and model
        model_params = dict(self.model.named_parameters())
        identity_matches = 0
        for proxy_name, proxy_param in self.proxy_params.items():
            if proxy_name in self.param_metadata:
                original_name, _, _ = self.param_metadata[proxy_name]
                if original_name in model_params:
                    model_param = model_params[original_name]
                    if id(proxy_param) == id(model_param):
                        identity_matches += 1
                    else:
                        logger.error(f"IDENTITY MISMATCH: {original_name} proxy_id={id(proxy_param)} vs model_id={id(model_param)}")
        
        logger.info(f"Parameter identity verification: {identity_matches}/{len(self.proxy_params)} parameters have matching IDs")
        
        # Update debug state if debugger is attached
        if hasattr(self, '_update_debug_state'):
            try:
                self._update_debug_state()
            except Exception as e:
                logger.debug(f"Failed to update debug state: {e}")
    
    def setup_block_swapping_compatibility(self, blocks_to_swap: int, device: torch.device):
        """
        Setup proxy parameters for block swapping compatibility.
        
        When block swapping is enabled, the last N blocks are moved to CPU during training.
        We need to ensure proxy parameters for those blocks are also on CPU for gradient flow.
        
        Args:
            blocks_to_swap: Number of blocks that will be swapped to CPU
            device: Training device (GPU)
        """
        if self.proxy_params is None or blocks_to_swap <= 0:
            return
            
        logger.info(f"Setting up selective fine-tuning compatibility with {blocks_to_swap} block swapping")
        
        # Find total number of blocks in the model by examining model structure
        total_blocks = 0
        for name, _ in self.model.named_parameters():
            if 'blocks.' in name:
                try:
                    block_parts = name.split('.')
                    if len(block_parts) >= 2 and block_parts[0] == 'blocks':
                        block_num = int(block_parts[1])
                        total_blocks = max(total_blocks, block_num + 1)
                except (ValueError, IndexError):
                    continue
                    
        if total_blocks == 0:
            logger.warning("Could not determine total blocks from model, using default of 40")
            total_blocks = 40  # Default for Wan DiT
        else:
            logger.info(f"Detected {total_blocks} total blocks in model")
        
        # Calculate which blocks will be swapped (last blocks_to_swap blocks)
        swapped_block_start = total_blocks - blocks_to_swap
        logger.info(f"Blocks {swapped_block_start} to {total_blocks-1} will be swapped to CPU")
        
        # Move proxy parameters to appropriate devices
        cpu_moved = 0
        gpu_moved = 0
        
        for proxy_name, proxy_param in self.proxy_params.items():
            if proxy_name in self.param_metadata:
                original_name, _, original_dtype = self.param_metadata[proxy_name]
                
                target_device = device  # Default to GPU
                
                # Check if this parameter belongs to a block that will be swapped
                if 'blocks.' in original_name:
                    # Extract block number (e.g., "blocks.25.self_attn.q.weight" -> 25)
                    try:
                        block_parts = original_name.split('.')
                        if len(block_parts) >= 2 and block_parts[0] == 'blocks':
                            block_num = int(block_parts[1])
                            
                            # Check if this block will be swapped to CPU
                            if block_num >= swapped_block_start:
                                target_device = torch.device('cpu')
                                cpu_moved += 1
                                logger.debug(f"Will move proxy {proxy_name} to CPU (block {block_num} will be swapped)")
                            else:
                                gpu_moved += 1
                                
                    except (ValueError, IndexError):
                        # If we can't parse block number, keep on GPU as fallback
                        gpu_moved += 1
                        logger.debug(f"Could not parse block number for {original_name}, keeping on GPU")
                else:
                    # Non-block parameters (embeddings, head) stay on GPU
                    gpu_moved += 1
                
                # FINAL FIX: Move data in-place to preserve parameter object identity
                logger.debug(f"[BLOCK SWAP IDENTITY FIX] {proxy_name}: before_id={id(proxy_param)}, target_device={target_device}")
                
                # Move data in-place (this preserves the Parameter object identity)
                proxy_param.data = proxy_param.data.to(device=target_device, dtype=original_dtype)
                proxy_param.requires_grad = True  # Ensure gradient computation
                
                # Verify the Parameter object ID didn't change
                logger.debug(f"[BLOCK SWAP IDENTITY FIX] {proxy_name}: after_id={id(proxy_param)}, device={proxy_param.device}")
                
                # Update in model - but the Parameter object is the same!
                _replace_model_parameter(self.model, original_name, proxy_param)
        
        # PHASE 2 FIX: Ensure all parameters still have requires_grad=True after block swapping setup
        self._ensure_proxy_params_require_grad("after_block_swapping_setup")
        
        logger.info(f"Block swapping compatibility setup complete: {cpu_moved} proxy params on CPU, {gpu_moved} on GPU")
    
    def ensure_gradient_device_consistency(self):
        """
        GRADIENT DEVICE FIX: Ensure gradients are on the same device as their parameters.
        This is critical for block swapping where parameters can be on CPU but gradients on GPU.
        Enhanced version that also handles optimizer state device consistency.
        """
        if self.proxy_params is None:
            return 0
            
        logger.debug("=== GRADIENT DEVICE FIX: Ensuring gradient-parameter device consistency ===")
        fixed_gradients = 0
        device_mismatches = []
        
        trainable_params = self.get_trainable_parameters()
        for i, param in enumerate(trainable_params):
            if param.grad is not None:
                param_device = param.device
                grad_device = param.grad.device
                
                if param_device != grad_device:
                    logger.debug(f"Fixing gradient device mismatch for param {i}: param on {param_device}, grad on {grad_device}")
                    device_mismatches.append((i, param_device, grad_device))
                    
                    # Fix the gradient device - clone to ensure it's properly moved
                    param.grad = param.grad.detach().to(param_device).clone()
                    fixed_gradients += 1
                    
                    # Verify the fix worked
                    if param.grad.device != param_device:
                        logger.error(f"CRITICAL: Failed to fix gradient device for param {i}!")
        
        if fixed_gradients > 0:
            logger.debug(f"Fixed gradient device mismatches for {fixed_gradients} parameters")
            for i, param_dev, grad_dev in device_mismatches[:3]:  # Log first few
                logger.debug(f"  Param {i}: {grad_dev} -> {param_dev}")
        
        return fixed_gradients
    
    def prepare_for_optimizer(self):
        """
        PHASE 3 FIX: Final preparation before optimizer creation.
        This MUST be called right before the optimizer is created to ensure
        all parameters have the correct requires_grad state.
        """
        logger.info("=== PHASE 3 FIX: Preparing parameters for optimizer creation ===")
        
        # Ensure all proxy parameters have requires_grad=True
        self._ensure_proxy_params_require_grad("before_optimizer_creation")
        
        # Get the final trainable parameters that will be given to the optimizer
        trainable_params = self.get_trainable_parameters()
        
        # Verify every parameter has requires_grad=True
        params_without_grad = []
        for i, param in enumerate(trainable_params):
            if not param.requires_grad:
                params_without_grad.append(i)
        
        if params_without_grad:
            logger.error(f"=== CRITICAL ERROR: {len(params_without_grad)} parameters still have requires_grad=False ===")
            logger.error(f"Parameter indices: {params_without_grad}")
            # Force fix them
            for i in params_without_grad:
                param = trainable_params[i]
                logger.error(f"Force-fixing parameter {i}: requires_grad=False -> True")
                param.requires_grad = True
        else:
            logger.info(f"✓ All {len(trainable_params)} parameters ready for optimizer (requires_grad=True)")
        
        # FINAL FIX: Verify parameter object identities before giving to optimizer
        logger.info("=== FINAL FIX: Verifying parameter identity consistency before optimizer ===")
        model_params = dict(self.model.named_parameters())
        identity_matches = 0
        identity_mismatches = []
        
        for i, param in enumerate(trainable_params):
            proxy_name = f"proxy_{i:06d}"
            if proxy_name in self.param_metadata:
                original_name, _, _ = self.param_metadata[proxy_name]
                if proxy_name in self.proxy_params:
                    proxy_param = self.proxy_params[proxy_name]
                    if id(param) == id(proxy_param):
                        identity_matches += 1
                    else:
                        identity_mismatches.append((i, proxy_name, original_name, id(param), id(proxy_param)))
                        logger.error(f"CRITICAL: Parameter {i} identity mismatch: model_id={id(param)} vs proxy_id={id(proxy_param)}")
        
        if identity_mismatches:
            logger.error(f"=== CRITICAL ERROR: {len(identity_mismatches)} parameter identity mismatches detected ===")
            logger.error("This will prevent optimizer updates from working!")
        else:
            logger.info(f"✓ Perfect parameter identity consistency: {identity_matches}/{len(trainable_params)} parameters match")
        
        return trainable_params
    
    def apply_updates_to_model(self):
        """Apply trained proxy parameter updates back to the original model."""
        if self.proxy_params is None or self.param_metadata is None:
            raise RuntimeError("Selective fine-tuning not initialized")
            
        apply_proxy_updates_to_model(
            self.model, self.proxy_params, self.param_metadata
        )
    
    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get memory usage information for monitoring.
        
        Returns:
            Dictionary containing memory usage statistics
        """
        if self.proxy_params is None:
            return {"status": "not_initialized"}
        
        proxy_param_count = sum(p.numel() for p in self.proxy_params.values())
        total_model_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            "status": "initialized",
            "proxy_parameters": len(self.proxy_params),
            "proxy_param_count": proxy_param_count,
            "total_model_params": total_model_params,
            "fraction_actual": proxy_param_count / total_model_params,
            "selected_param_names": self.selected_param_names[:10] + ["..."] if len(self.selected_param_names) > 10 else self.selected_param_names
        }
    
    def save_selective_state(self) -> Dict[str, Any]:
        """
        Save the selective fine-tuning state for checkpointing.
        
        Returns:
            State dictionary containing selective fine-tuning information
        """
        if self.proxy_params is None:
            raise RuntimeError("Selective fine-tuning not initialized")
        
        return {
            "fraction": self.fraction,
            "selection_id": self.selection_id,
            "selected_param_names": self.selected_param_names,
            "param_metadata": self.param_metadata,
            "proxy_params": self.proxy_params.state_dict()
        }
    
    def load_selective_state(self, state_dict: Dict[str, Any]):
        """
        Load selective fine-tuning state from checkpoint.
        
        Args:
            state_dict: State dictionary from save_selective_state()
        """
        self.fraction = state_dict["fraction"]
        self.selection_id = state_dict["selection_id"]
        self.selected_param_names = state_dict["selected_param_names"]
        self.param_metadata = state_dict["param_metadata"]
        
        # Recreate proxy parameters structure
        self.proxy_params = nn.ParameterDict()
        self.proxy_params.load_state_dict(state_dict["proxy_params"])
        
        logger.info(f"Loaded selective fine-tuning state: "
                   f"fraction={self.fraction:.6f}, selection_id={self.selection_id}")


def is_selective_finetuning_enabled(args) -> bool:
    """
    Check if selective fine-tuning is enabled based on arguments.
    
    Args:
        args: Argument namespace
        
    Returns:
        True if selective fine-tuning should be used
    """
    return hasattr(args, 'ff') and args.ff is not None and args.ff > 0


def validate_selective_finetuning_args(args):
    """
    Validate selective fine-tuning arguments.
    
    Args:
        args: Argument namespace
        
    Raises:
        ValueError: If arguments are invalid
    """
    if not hasattr(args, 'ff') or args.ff is None:
        raise ValueError("--ff argument is required for selective fine-tuning")
    
    if not hasattr(args, 'ffid') or args.ffid is None:
        raise ValueError("--ffid argument is required for selective fine-tuning")
    
    if args.ff <= 0 or args.ff > 1:
        raise ValueError(f"--ff must be between 0 and 1, got {args.ff}")
    
    if args.ffid < 0:
        raise ValueError(f"--ffid must be non-negative, got {args.ffid}")
    
    logger.info(f"Validated selective fine-tuning args: ff={args.ff:.6f}, ffid={args.ffid}")


class SelectiveFinetuneWrapper(nn.Module):
    """
    Wrapper that makes a SelectiveFinetuneManager look like a standard model
    for compatibility with the existing training infrastructure.
    """
    
    def __init__(self, model: nn.Module, selective_finetuner: SelectiveFinetuneManager):
        super().__init__()
        self.model = model
        self.selective_finetuner = selective_finetuner
        
        # Copy important attributes from the original model
        if hasattr(model, 'gradient_checkpointing'):
            self.gradient_checkpointing = model.gradient_checkpointing
    
    def forward(self, *args, **kwargs):
        """
        Forward pass that handles CPU<->GPU data movement automatically.
        The model already contains proxy parameters, so gradients flow directly to them.
        """
        # Move all tensor inputs to CPU (where the main model is)
        cpu_args = []
        original_device = None
        
        for arg in args:
            if isinstance(arg, torch.Tensor):
                if original_device is None:
                    original_device = arg.device
                cpu_args.append(arg.cpu())
            elif isinstance(arg, list):
                cpu_arg_list = []
                for item in arg:
                    if isinstance(item, torch.Tensor):
                        if original_device is None:
                            original_device = item.device
                        cpu_arg_list.append(item.cpu())
                    else:
                        cpu_arg_list.append(item)
                cpu_args.append(cpu_arg_list)
            else:
                cpu_args.append(arg)
        
        cpu_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                if original_device is None:
                    original_device = value.device
                cpu_kwargs[key] = value.cpu()
            elif isinstance(value, list):
                cpu_value_list = []
                for item in value:
                    if isinstance(item, torch.Tensor):
                        if original_device is None:
                            original_device = item.device
                        cpu_value_list.append(item.cpu())
                    else:
                        cpu_value_list.append(item)
                cpu_kwargs[key] = cpu_value_list
            else:
                cpu_kwargs[key] = value
        
        # Forward pass on CPU (model now contains proxy parameters directly)
        output = self.model(*cpu_args, **cpu_kwargs)
        
        # Move output back to original device
        if original_device is not None and original_device.type != 'cpu':
            if isinstance(output, torch.Tensor):
                output = output.to(original_device)
            elif isinstance(output, list):
                output = [item.to(original_device) if isinstance(item, torch.Tensor) else item for item in output]
        
        return output
    
    def parameters(self, recurse: bool = True):
        """Return only the proxy parameters that are now in the model for training."""
        if self.selective_finetuner.proxy_params is None or self.selective_finetuner.param_metadata is None:
            return iter([])
        
        # Return the proxy parameters that are now actually in the model
        model_params = dict(self.model.named_parameters())
        selected_params = []
        
        for proxy_name in self.selective_finetuner.param_metadata:
            original_name, _, _ = self.selective_finetuner.param_metadata[proxy_name]
            if original_name in model_params:
                selected_params.append(model_params[original_name])
        
        return iter(selected_params)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """Return named proxy parameters."""
        proxy_params = self.selective_finetuner.proxy_params
        if proxy_params is not None:
            for name, param in proxy_params.items():
                yield name, param
    
    def state_dict(self, *args, **kwargs):
        """Return main model state dict."""
        return self.model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        """Load state dict into main model."""
        return self.model.load_state_dict(*args, **kwargs)
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self.model.train(mode)
        return super().train(mode)
    
    def eval(self):
        """Set evaluation mode."""
        return self.train(False)
    
    def to(self, device_or_dtype, **kwargs):
        """
        Handle device/dtype changes. Keep main model on CPU, move proxy params to device.
        """
        logger.info(f"=== WRAPPER .to() CALLED with {device_or_dtype} ===")
        if isinstance(device_or_dtype, (torch.device, str)):
            device = torch.device(device_or_dtype) if isinstance(device_or_dtype, str) else device_or_dtype
            if device.type != 'cpu':
                # Move proxy parameters to the device, keep main model on CPU
                logger.info(f"Moving selective fine-tuning parameters to {device}")
                self.selective_finetuner.move_proxy_params_to_device(device)
                
                # CRITICAL DEBUG: Check if parameters are still the same after moving
                trainable_params = self.selective_finetuner.get_trainable_parameters()
                logger.info(f"After .to(), first 3 param IDs: {[id(p) for p in trainable_params[:3]]}")
            return self
        else:
            # Handle dtype conversion
            return super().to(device_or_dtype, **kwargs)
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on the main model."""
        if hasattr(self.model, 'enable_gradient_checkpointing'):
            self.model.enable_gradient_checkpointing()
            self.gradient_checkpointing = True
            logger.info("Enabled gradient checkpointing for selective fine-tuning")
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing on the main model."""
        if hasattr(self.model, 'disable_gradient_checkpointing'):
            self.model.disable_gradient_checkpointing()
            self.gradient_checkpointing = False
            logger.info("Disabled gradient checkpointing for selective fine-tuning")
    
    def apply_updates(self):
        """Apply proxy parameter updates to the main model."""
        self.selective_finetuner.apply_updates_to_model()
    
    def _validate_updates_applied(self):
        """Validate that proxy parameter updates were properly applied to main model."""
        if not hasattr(self, 'selective_finetuner') or self.selective_finetuner.proxy_params is None:
            return True
        
        model_params = dict(self.model.named_parameters())
        mismatches = 0
        total_checked = 0
        
        for proxy_name, proxy_param in self.selective_finetuner.proxy_params.items():
            if proxy_name in self.selective_finetuner.param_metadata:
                original_name, _, _ = self.selective_finetuner.param_metadata[proxy_name]
                if original_name in model_params:
                    model_param = model_params[original_name]
                    total_checked += 1
                    
                    # Compare parameters on CPU to avoid device issues
                    proxy_data_cpu = proxy_param.data.detach().cpu().float()
                    model_data_cpu = model_param.data.detach().cpu().float()
                    
                    if not torch.allclose(proxy_data_cpu, model_data_cpu, rtol=1e-5, atol=1e-6):
                        mismatches += 1
                        if mismatches <= 3:  # Log first few mismatches
                            mean_diff = (proxy_data_cpu - model_data_cpu).abs().mean().item()
                            logger.warning(f"Parameter mismatch {original_name}: mean_diff={mean_diff:.8f}")
        
        if mismatches > 0:
            logger.warning(f"Detected {mismatches}/{total_checked} parameter mismatches - updates may not be fully applied!")
            return False
        else:
            logger.info(f"✓ All {total_checked} proxy parameter updates properly applied to model")
            return True
    
    def _apply_updates_with_verification(self):
        """Apply updates with comprehensive verification and logging."""
        logger.info("Applying proxy parameter updates with verification...")
        
        # Step 1: Apply updates using existing method
        self.apply_updates()
        
        # Step 2: Verify each parameter was updated correctly
        model_params = dict(self.model.named_parameters())
        successful_updates = 0
        failed_updates = 0
        
        for proxy_name, proxy_param in self.selective_finetuner.proxy_params.items():
            if proxy_name in self.selective_finetuner.param_metadata:
                original_name, _, _ = self.selective_finetuner.param_metadata[proxy_name]
                if original_name in model_params:
                    model_param = model_params[original_name]
                    
                    # Verify the update was applied
                    proxy_data = proxy_param.data.detach().cpu().float()
                    model_data = model_param.data.detach().cpu().float()
                    
                    if torch.allclose(proxy_data, model_data, rtol=1e-5, atol=1e-6):
                        successful_updates += 1
                    else:
                        failed_updates += 1
                        logger.warning(f"Update verification failed for {original_name}")
        
        logger.info(f"Update verification: {successful_updates} successful, {failed_updates} failed")
        
        if failed_updates > 0:
            logger.warning(f"{failed_updates} parameter updates may not have been applied correctly")
    
    def _force_update_application(self):
        """
        Force application of proxy parameter updates with enhanced error handling.
        This is a backup method when standard update application fails.
        """
        logger.info("Forcing proxy parameter updates with enhanced verification...")
        
        model_params = dict(self.model.named_parameters())
        force_updated = 0
        
        for proxy_name, proxy_param in self.selective_finetuner.proxy_params.items():
            if proxy_name not in self.selective_finetuner.param_metadata:
                continue
                
            original_name, expected_shape, expected_dtype = self.selective_finetuner.param_metadata[proxy_name]
            
            if original_name not in model_params:
                logger.warning(f"Cannot force update {original_name} - parameter not found in model")
                continue
                
            original_param = model_params[original_name]
            
            # Validate shapes match
            if proxy_param.shape != expected_shape or proxy_param.shape != original_param.shape:
                logger.error(f"Shape mismatch preventing update for {original_name}: "
                           f"proxy={proxy_param.shape}, expected={expected_shape}, model={original_param.shape}")
                continue
            
            # Force the update with comprehensive device and dtype handling
            try:
                with torch.no_grad():
                    # Ensure proxy data is moved to correct device/dtype
                    updated_data = proxy_param.data.to(
                        device=original_param.device,
                        dtype=original_param.dtype
                    )
                    
                    # Force copy the data
                    original_param.data.copy_(updated_data)
                    
                    # Verify the copy worked
                    if torch.allclose(original_param.data.cpu().float(), proxy_param.data.cpu().float(), rtol=1e-5, atol=1e-6):
                        force_updated += 1
                        logger.debug(f"Force updated {original_name} successfully")
                    else:
                        logger.error(f"Force update failed for {original_name} - data did not copy correctly")
                        
            except Exception as e:
                logger.error(f"Exception during force update of {original_name}: {e}")
        
        logger.info(f"Force update completed: {force_updated} parameters updated")
    
    def _verify_state_dict_contains_updates(self, state_dict: Dict[str, torch.Tensor]) -> bool:
        """
        Verify that the state dict contains the trained values from proxy parameters.
        This is the final check before saving to ensure we're not saving untrained weights.
        """
        logger.info("Verifying state dict contains trained parameter values...")
        
        verified_count = 0
        mismatch_count = 0
        
        for proxy_name, proxy_param in self.selective_finetuner.proxy_params.items():
            if proxy_name not in self.selective_finetuner.param_metadata:
                continue
                
            original_name, _, _ = self.selective_finetuner.param_metadata[proxy_name]
            
            if original_name not in state_dict:
                logger.warning(f"Parameter {original_name} not found in state dict")
                continue
            
            state_dict_param = state_dict[original_name]
            
            # Compare proxy parameter with state dict parameter
            proxy_data_cpu = proxy_param.data.detach().cpu().float()
            state_dict_data_cpu = state_dict_param.detach().cpu().float()
            
            if torch.allclose(proxy_data_cpu, state_dict_data_cpu, rtol=1e-5, atol=1e-6):
                verified_count += 1
            else:
                mismatch_count += 1
                mean_diff = (proxy_data_cpu - state_dict_data_cpu).abs().mean().item()
                logger.warning(f"State dict verification FAILED for {original_name}: mean_diff={mean_diff:.8f}")
                
                if mismatch_count <= 3:  # Log details for first few failures
                    max_diff = (proxy_data_cpu - state_dict_data_cpu).abs().max().item()
                    logger.warning(f"  Details: proxy_mean={proxy_data_cpu.mean():.6f}, "
                                 f"state_dict_mean={state_dict_data_cpu.mean():.6f}, "
                                 f"max_diff={max_diff:.8f}")
        
        logger.info(f"State dict verification: {verified_count} verified, {mismatch_count} mismatched")
        
        if mismatch_count > 0:
            logger.error(f"CRITICAL: {mismatch_count} parameters in state dict do not match trained proxy values!")
            logger.error("This means the saved model will NOT contain the trained weights!")
            return False
        else:
            logger.info(f"✓ All {verified_count} trained parameters verified in state dict")
            return True
        
    def save_weights(self, filepath: str, dtype: Optional[torch.dtype] = None, metadata: Optional[Dict] = None):
        """
        Enhanced save_weights with robust proxy update application and comprehensive validation.
        Ensures trained proxy parameters are permanently merged into the saved model.
        """
        logger.info("=== SAVING MODEL WITH SELECTIVE FINE-TUNING ===")
        logger.info("Applying proxy parameter updates for permanent model saving...")
        
        try:
            # STEP 1: Force comprehensive update application
            logger.info("Step 1: Applying proxy parameter updates to main model...")
            self._apply_updates_with_verification()
            
            # STEP 2: Comprehensive validation
            logger.info("Step 2: Validating all updates were applied correctly...")
            updates_valid = self._validate_updates_applied()
            if not updates_valid:
                logger.error("CRITICAL: Some proxy updates were not applied correctly!")
                # Try applying updates again with extra verification
                logger.info("Attempting secondary update application...")
                self._force_update_application()
                updates_valid = self._validate_updates_applied()
                
                if not updates_valid:
                    raise RuntimeError("Failed to apply proxy parameter updates to model - cannot save correctly trained model")
            
            # STEP 3: Additional state dict verification
            logger.info("Step 3: Performing final state dict verification...")
            state_dict = self.model.state_dict()
            final_verification = self._verify_state_dict_contains_updates(state_dict)
            
            if not final_verification:
                logger.error("CRITICAL: State dict does not contain expected trained values!")
                raise RuntimeError("State dict verification failed - trained values not found in model")
            
            # STEP 4: Convert dtype if specified
            if dtype is not None:
                logger.info(f"Step 4: Converting state dict to dtype {dtype}...")
                for key in state_dict:
                    if state_dict[key].dtype.is_floating_point:
                        state_dict[key] = state_dict[key].to(dtype)
            
            # STEP 5: Save the model
            logger.info(f"Step 5: Saving model to {filepath}...")
            if filepath.endswith('.safetensors'):
                from safetensors.torch import save_file
                
                if metadata is None:
                    metadata = {}
                
                # Add selective fine-tuning metadata (simplified as requested)
                metadata.update({
                    'selective_finetune_applied': 'true',
                    'trained_params': str(len(self.selective_finetuner.proxy_params))
                })
                
                save_file(state_dict, filepath, metadata)
            else:
                torch.save(state_dict, filepath)
            
            logger.info(f"✓ Successfully saved selective fine-tuned model to: {filepath}")
            logger.info(f"✓ Model contains {len(self.selective_finetuner.proxy_params)} trained parameters")
            logger.info(f"✓ Saved model should work perfectly with standard inference scripts")
            logger.info("=== SAVE COMPLETE ===")
            
        except Exception as e:
            logger.error(f"Error during selective fine-tuning model save: {e}")
            logger.error("This may result in saved models that don't match training quality!")
            raise
    
    @property
    def dtype(self):
        """Return the dtype of the main model."""
        return self.model.dtype
    
    @property
    def device(self):
        """Return the device of the main model."""
        return self.model.device
    
    def prepare_optimizer_params(self, unet_lr=None, text_encoder_lr=None, **kwargs):
        """
        Prepare optimizer parameters for selective fine-tuning.
        Returns proxy parameters with specified learning rate.
        """
        # PHASE 3 FIX: Prepare parameters for optimizer (ensures requires_grad=True)
        proxy_params = self.selective_finetuner.prepare_for_optimizer()
        
        # CRITICAL DEBUG: Log what we're giving to the optimizer
        logger.info(f"=== OPTIMIZER SETUP DEBUG ===")
        logger.info(f"Preparing {len(proxy_params)} parameters for optimizer")
        for i, param in enumerate(proxy_params[:3]):
            logger.info(f"  Param {i}: id={id(param)}, requires_grad={param.requires_grad}, device={param.device}, shape={param.shape}")
        logger.info(f"=== END OPTIMIZER SETUP ===")
        
        # Create parameter groups for the optimizer
        trainable_params = []
        lr_descriptions = []
        
        if proxy_params:
            lr = unet_lr if unet_lr is not None else 1e-4  # fallback learning rate
            trainable_params.append({
                "params": proxy_params,
                "lr": lr
            })
            lr_descriptions.append(f"selective_finetune")
        
        return trainable_params, lr_descriptions
    
    def prepare_grad_etc(self, transformer):
        """
        Prepare gradient-related setup for selective fine-tuning.
        For selective fine-tuning, we don't need special gradient preparation
        since we're working with standard torch parameters (just fewer of them).
        """
        # No special gradient preparation needed for selective fine-tuning
        pass
    
    def on_epoch_start(self, transformer):
        """Called at the start of each epoch."""
        # For selective fine-tuning, no special epoch start handling needed
        pass
    
    def on_step_start(self):
        """Called at the start of each training step."""
        # For selective fine-tuning, no special step start handling needed  
        pass
    
    def get_trainable_params(self):
        """Return trainable proxy parameters for gradient clipping."""
        params = self.selective_finetuner.get_trainable_parameters()
        logger.debug(f"Wrapper returning {len(params)} trainable params for gradient clipping")
        return params
    
    def apply_max_norm_regularization(self, max_norm, device):
        """Apply max norm regularization to proxy parameters."""
        # For selective fine-tuning, we can apply standard max norm to proxy parameters
        # This is a simplified implementation - return None values to indicate no scaling was done
        return None, None, None
    
    def apply_to(self, text_encoder, unet, apply_text_encoder=False, apply_unet=True):
        """
        Apply method expected by the training infrastructure.
        For selective fine-tuning, this is a no-op since the wrapper already contains everything.
        """
        logger.info("apply_to called for selective fine-tuning - no action needed")
        pass
    
    def load_weights(self, weights_path):
        """
        Load weights method expected by training infrastructure.
        For selective fine-tuning, we could implement loading proxy parameters.
        """
        logger.info(f"load_weights called for selective fine-tuning: {weights_path}")
        # For now, return empty info since we're not implementing weight loading
        return "selective fine-tuning weights not loaded (not implemented)"
    
    def prepare_network(self, args):
        """
        Prepare network method expected by training infrastructure.
        For selective fine-tuning, this is a no-op since everything is already set up.
        """
        logger.info("prepare_network called for selective fine-tuning - no action needed")
        pass