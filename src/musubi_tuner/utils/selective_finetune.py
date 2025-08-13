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
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict

logger = logging.getLogger(__name__)


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
    selection_id: int
) -> List[str]:
    """
    Deterministically select a fraction of parameters evenly distributed across the model.
    Selection is based on total parameter count, not number of tensors.
    
    Args:
        all_params: Dictionary of all available parameters
        fraction: Fraction of parameters to select (e.g., 0.00001)
        selection_id: ID for deterministic selection (seed)
        
    Returns:
        List of selected parameter names
    """
    # Set random seed for deterministic selection
    original_state = random.getstate()
    random.seed(selection_id)
    
    try:
        # Calculate total parameter count across all tensors
        total_model_params = sum(param.numel() for param in all_params.values())
        target_param_count = max(1, int(total_model_params * fraction))
        
        logger.info(f"Target: {target_param_count:,} parameters out of {total_model_params:,} "
                   f"(fraction={fraction:.6f}, selection_id={selection_id})")
        
        # Create list of (param_name, param_count) sorted by count
        param_info = [(name, param.numel()) for name, param in all_params.items()]
        param_info.sort(key=lambda x: x[1])  # Sort by parameter count (smallest first)
        
        # Group parameters by component type for even distribution across ALL components
        # CRITICAL: Individual blocks must be separate groups to ensure even distribution
        # across early blocks (character) and later blocks (style) as specified in CLAUDE.md
        param_groups = {
            'patch_embedding': [],
            'text_embedding': [],
            'time_embedding': [],
            'time_projection': [],
            'head': [],
            'other': []
        }
        
        # Add individual block groups (blocks.0, blocks.1, ..., blocks.39)
        for i in range(40):  # Wan DiT has 40 blocks
            param_groups[f'blocks.{i}'] = []
        
        for name, count in param_info:
            if 'patch_embedding' in name:
                param_groups['patch_embedding'].append((name, count))
            elif 'text_embedding' in name:
                param_groups['text_embedding'].append((name, count))
            elif 'time_embedding' in name:
                param_groups['time_embedding'].append((name, count))
            elif 'time_projection' in name:
                param_groups['time_projection'].append((name, count))
            elif 'blocks.' in name:
                # Extract block number and assign to individual block group
                try:
                    block_parts = name.split('.')
                    if len(block_parts) >= 2 and block_parts[0] == 'blocks':
                        block_num = int(block_parts[1])
                        if 0 <= block_num < 40:
                            param_groups[f'blocks.{block_num}'].append((name, count))
                        else:
                            param_groups['other'].append((name, count))
                    else:
                        param_groups['other'].append((name, count))
                except (ValueError, IndexError):
                    param_groups['other'].append((name, count))
            elif 'head.' in name:
                param_groups['head'].append((name, count))
            else:
                param_groups['other'].append((name, count))
        
        # MATHEMATICALLY CORRECT STRATEGY: Guarantee all 40 blocks coverage first
        # Phase 1: Select smallest parameter from each block (guaranteed coverage)
        # Phase 2: Use remaining budget for additional parameters across all blocks
        
        selected_params = []
        selected_param_count = 0
        
        logger.info("=== PHASE 1: Guaranteeing coverage of ALL 40 blocks ===")
        
        # Step 1: Identify all block groups and ensure we cover ALL 40 blocks
        block_groups = {gname: gparams for gname, gparams in param_groups.items() 
                       if gname.startswith('blocks.')}
        non_block_groups = {gname: gparams for gname, gparams in param_groups.items() 
                           if not gname.startswith('blocks.') and gparams}
        
        logger.info(f"Found {len(block_groups)} block groups and {len(non_block_groups)} non-block groups")
        
        # Phase 1: Select ONE smallest parameter from each block (guaranteed 40-block coverage)
        covered_blocks = set()
        phase1_budget = target_param_count // 2  # Reserve half budget for guaranteed coverage
        
        # Process block groups in deterministic order
        for group_name in sorted(block_groups.keys()):
            group_params = block_groups[group_name]
            if not group_params:
                continue
                
            # Sort by size (smallest first) for guaranteed fit
            group_params.sort(key=lambda x: x[1])
            
            # Use selection_id for deterministic offset within each block
            block_num = int(group_name.split('.')[1])
            param_offset = (selection_id + block_num) % len(group_params)
            
            # Select the parameter at the offset, or smallest if it doesn't fit
            selected_param = None
            for attempt in range(len(group_params)):
                idx = (param_offset + attempt) % len(group_params)
                param_name, param_count = group_params[idx]
                
                if selected_param_count + param_count <= phase1_budget:
                    selected_param = (param_name, param_count)
                    break
            
            # If no parameter fits in phase1_budget, take the smallest anyway (essential for coverage)
            if selected_param is None and group_params:
                selected_param = group_params[0]  # Smallest parameter
                logger.warning(f"Phase 1 budget exceeded, taking smallest param from {group_name}")
            
            if selected_param:
                param_name, param_count = selected_param
                selected_params.append(selected_param)
                selected_param_count += param_count
                covered_blocks.add(block_num)
                logger.info(f"Phase 1: Selected {param_name} from {group_name}: {param_count:,} parameters")
        
        logger.info(f"Phase 1 complete: Covered {len(covered_blocks)} blocks with {selected_param_count:,} parameters")
        
        # Step 2: Use remaining budget for non-block components and additional block parameters
        logger.info("=== PHASE 2: Using remaining budget for additional parameters ===")
        remaining_budget = target_param_count - selected_param_count
        
        if remaining_budget > 0:
            # Collect all unselected parameters
            all_unselected = []
            
            # Add non-block parameters
            for group_name, group_params in non_block_groups.items():
                for param_name, param_count in group_params:
                    if param_name not in [p[0] for p in selected_params]:
                        all_unselected.append((param_name, param_count, group_name))
            
            # Add remaining block parameters
            for group_name, group_params in block_groups.items():
                for param_name, param_count in group_params:
                    if param_name not in [p[0] for p in selected_params]:
                        all_unselected.append((param_name, param_count, group_name))
            
            # Sort deterministically for consistent selection
            all_unselected.sort(key=lambda x: x[0])
            
            # Use selection_id to determine starting point for phase 2
            if all_unselected:
                start_idx = selection_id % len(all_unselected)
                
                # Select parameters in round-robin fashion until budget exhausted
                for i in range(len(all_unselected)):
                    idx = (start_idx + i) % len(all_unselected)
                    param_name, param_count, group_name = all_unselected[idx]
                    
                    if param_count <= remaining_budget:
                        selected_params.append((param_name, param_count))
                        selected_param_count += param_count
                        remaining_budget -= param_count
                        logger.info(f"Phase 2: Selected {param_name} from {group_name}: {param_count:,} parameters")
                    
                    if remaining_budget <= 0:
                        break
        
        logger.info(f"Phase 2 complete: Total selected {selected_param_count:,} parameters")
        logger.info(f"Block coverage verification: {len(covered_blocks)}/40 blocks covered")
        
        if len(covered_blocks) < 40:
            # Log which blocks are missing
            missing_blocks = set(range(40)) - covered_blocks
            logger.warning(f"Missing blocks: {sorted(missing_blocks)}")
        else:
            logger.info("✅ Perfect coverage: ALL 40 blocks covered!")
        
        # Extract just the parameter names
        selected_param_names = [name for name, count in selected_params]
        
        actual_fraction = selected_param_count / total_model_params
        
        # Log distribution across individual blocks to verify even distribution
        block_distribution = {}
        component_distribution = {}
        
        for name, count in selected_params:
            if 'blocks.' in name:
                try:
                    block_parts = name.split('.')
                    if len(block_parts) >= 2 and block_parts[0] == 'blocks':
                        block_num = int(block_parts[1])
                        block_key = f"block_{block_num}"
                        block_distribution[block_key] = block_distribution.get(block_key, 0) + count
                except (ValueError, IndexError):
                    pass
            
            # Track component distribution
            if 'patch_embedding' in name:
                component_distribution['patch_embedding'] = component_distribution.get('patch_embedding', 0) + count
            elif 'text_embedding' in name:
                component_distribution['text_embedding'] = component_distribution.get('text_embedding', 0) + count
            elif 'time_embedding' in name:
                component_distribution['time_embedding'] = component_distribution.get('time_embedding', 0) + count
            elif 'time_projection' in name:
                component_distribution['time_projection'] = component_distribution.get('time_projection', 0) + count
            elif 'head.' in name:
                component_distribution['head'] = component_distribution.get('head', 0) + count
            elif 'blocks.' in name:
                component_distribution['blocks'] = component_distribution.get('blocks', 0) + count
            else:
                component_distribution['other'] = component_distribution.get('other', 0) + count
        
        logger.info(f"Final selection: {len(selected_param_names)} parameter tensors")
        logger.info(f"Total selected: {selected_param_count:,} parameters out of {total_model_params:,}")
        logger.info(f"Actual fraction: {actual_fraction:.6f} (target: {fraction:.6f})")
        logger.info(f"Difference: {abs(actual_fraction - fraction)/fraction*100:.1f}% from target")
        
        # Log component distribution
        logger.info("Parameter distribution by component:")
        for component, count in sorted(component_distribution.items()):
            percentage = (count / selected_param_count) * 100
            logger.info(f"  {component}: {count:,} parameters ({percentage:.1f}%)")
        
        # Log block distribution (only if we have block parameters)
        if block_distribution:
            block_count = len(block_distribution)
            logger.info(f"Parameter distribution across {block_count} blocks:")
            for block_key in sorted(block_distribution.keys(), key=lambda x: int(x.split('_')[1])):
                count = block_distribution[block_key]
                percentage = (count / selected_param_count) * 100
                logger.info(f"  {block_key}: {count:,} parameters ({percentage:.1f}%)")
        else:
            logger.info("No block parameters selected in this subset")
        
        return selected_param_names
        
    finally:
        # Restore original random state
        random.setstate(original_state)


def create_proxy_parameters(
    model: nn.Module, 
    selected_param_names: List[str]
) -> Tuple[nn.ParameterDict, Dict[str, Tuple[str, torch.Size, torch.dtype]]]:
    """
    Create proxy parameters for the selected subset and replace original parameters.
    
    Args:
        model: The model containing the original parameters
        selected_param_names: Names of parameters to create proxies for
        
    Returns:
        Tuple of (proxy_parameters, parameter_metadata)
        - proxy_parameters: ParameterDict containing proxy parameters
        - parameter_metadata: Metadata for mapping back to original parameters
    """
    proxy_params = nn.ParameterDict()
    param_metadata = {}
    
    model_params = dict(model.named_parameters())
    
    for i, param_name in enumerate(selected_param_names):
        if param_name not in model_params:
            logger.warning(f"Parameter {param_name} not found in model")
            continue
            
        original_param = model_params[param_name]
        
        # Create proxy parameter with same shape and dtype
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
        
        # Select subset deterministically
        self.selected_param_names = select_parameters_deterministic(
            all_params, self.fraction, self.selection_id
        )
        
        # Create proxy parameters
        self.proxy_params, self.param_metadata = create_proxy_parameters(
            self.model, self.selected_param_names
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
        
        for proxy_name in self.param_metadata:
            original_name, _, _ = self.param_metadata[proxy_name]
            if original_name in model_params:
                param = model_params[original_name]
                
                # CRITICAL FIX: Ensure the parameter has requires_grad=True
                if not param.requires_grad:
                    logger.warning(f"Parameter {original_name} had requires_grad=False, fixing to True")
                    param.requires_grad = True
                    requires_grad_fixes += 1
                
                trainable_params.append(param)
        
        # PHASE 1 DEBUG: Log summary of requires_grad fixes
        if requires_grad_fixes > 0:
            logger.warning(f"=== PHASE 1 DEBUG: Fixed requires_grad=False for {requires_grad_fixes}/{len(trainable_params)} parameters ===")
            logger.warning(f"This indicates parameters are losing requires_grad=True somewhere in the setup process")
        
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
        
        # Estimate memory usage (assuming float32 for simplicity)
        proxy_memory_mb = proxy_param_count * 4 / (1024 * 1024)
        
        return {
            "status": "initialized",
            "proxy_parameters": len(self.proxy_params),
            "proxy_param_count": proxy_param_count,
            "total_model_params": total_model_params,
            "fraction_actual": proxy_param_count / total_model_params,
            "estimated_proxy_memory_mb": proxy_memory_mb,
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
        
    def save_weights(self, filepath: str, dtype: Optional[torch.dtype] = None, metadata: Optional[Dict] = None):
        """
        Enhanced save_weights with complete proxy update application and validation.
        """
        logger.info("Applying proxy parameter updates for model saving...")
        
        # SAFETY: Store original proxy parameter devices for restoration
        original_devices = {}
        for proxy_name, proxy_param in self.selective_finetuner.proxy_params.items():
            original_devices[proxy_name] = proxy_param.device
        
        try:
            # Apply updates using existing method (preserves working logic)
            self.apply_updates()
            
            # VALIDATION: Verify updates were applied correctly
            updates_valid = self._validate_updates_applied()
            if not updates_valid:
                logger.warning("Some proxy updates may not have been applied correctly, but proceeding with save...")
            
            # Get state dict (should now have all updates applied)
            state_dict = self.model.state_dict()
            
            # Convert dtype if specified
            if dtype is not None:
                for key in state_dict:
                    if state_dict[key].dtype.is_floating_point:
                        state_dict[key] = state_dict[key].to(dtype)
            
            # Save using safetensors if filepath ends with .safetensors
            if filepath.endswith('.safetensors'):
                from safetensors.torch import save_file
                
                if metadata is None:
                    metadata = {}
                
                # Add selective fine-tuning metadata
                metadata.update({
                    'selective_finetune_fraction': str(self.selective_finetuner.fraction),
                    'selective_finetune_selection_id': str(self.selective_finetuner.selection_id),
                    'selective_finetune_proxy_params': str(len(self.selective_finetuner.proxy_params)),
                    'selective_finetune_updates_validated': str(updates_valid)
                })
                
                save_file(state_dict, filepath, metadata)
            else:
                torch.save(state_dict, filepath)
            
            logger.info(f"Saved selective fine-tuned model to: {filepath}")
            
        except Exception as e:
            logger.error(f"Error during selective fine-tuning model save: {e}")
            raise
        
        # Note: We don't restore original devices here because the proxy parameters
        # should remain where they are for continued training
    
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