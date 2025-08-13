"""
Minimal debugging utilities for selective fine-tuning training
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import hashlib

logger = logging.getLogger(__name__)


class SelectiveTrainingDebugger:
    """Minimal debugger for tracking gradient flow and weight changes in selective fine-tuning."""
    
    def __init__(self, selective_finetuner):
        self.selective_finetuner = selective_finetuner
        self.initial_param_hashes = {}
        self.step_count = 0
        self.last_gradient_info = {}
        
        # Store initial parameter hashes for change detection
        self._store_initial_param_hashes()
    
    def _store_initial_param_hashes(self):
        """Store hashes of initial parameter values from the model (after proxy replacement)."""
        if self.selective_finetuner.proxy_params is not None:
            # Get the trainable parameters from the model (which are now the proxy parameters)
            trainable_params = self.selective_finetuner.get_trainable_parameters()
            
            for i, param in enumerate(trainable_params):
                proxy_name = f"proxy_{i:06d}"
                # Convert to float32 first to handle BFloat16
                param_data = param.data.cpu().float()
                param_bytes = param_data.numpy().tobytes()
                param_hash = hashlib.md5(param_bytes).hexdigest()[:8]
                self.initial_param_hashes[proxy_name] = param_hash
                logger.debug(f"Initial hash for model param {i} ({proxy_name}): {param_hash}")
    
    def log_gradient_flow(self, loss: torch.Tensor, step: int):
        """Log minimal gradient flow information after backward pass."""
        self.step_count = step
        
        if self.selective_finetuner.proxy_params is None:
            logger.warning("No proxy parameters found for gradient debugging")
            return
        
        grad_info = {
            'with_grad': 0,
            'without_grad': 0,
            'total_grad_norm': 0.0,
            'max_grad': 0.0,
            'loss': loss.item()
        }
        
        # Check gradients on the parameters that are actually in the model now
        trainable_params = self.selective_finetuner.get_trainable_parameters()
        logger.debug(f"Checking gradients on {len(trainable_params)} trainable parameters from model")
        
        # CRITICAL DEBUG: Check if these parameters have requires_grad=True
        requires_grad_count = sum(1 for p in trainable_params if p.requires_grad)
        logger.debug(f"Parameters with requires_grad=True: {requires_grad_count}/{len(trainable_params)}")
        
        # CRITICAL DEBUG: Check device placement
        device_info = {}
        for i, param in enumerate(trainable_params):
            device_str = str(param.device)
            device_info[device_str] = device_info.get(device_str, 0) + 1
            
            if param.grad is not None:
                grad_info['with_grad'] += 1
                grad_norm = param.grad.data.norm().item()
                grad_info['total_grad_norm'] += grad_norm
                grad_info['max_grad'] = max(grad_info['max_grad'], grad_norm)
                if i < 3:  # Log first few for debugging
                    logger.debug(f"Param {i}: grad norm = {grad_norm:.8f}, device={param.device}, requires_grad={param.requires_grad}")
            else:
                grad_info['without_grad'] += 1
                if i < 3:  # Log first few for debugging
                    logger.debug(f"Param {i}: NO GRADIENT, device={param.device}, requires_grad={param.requires_grad}, is_leaf={param.is_leaf}")
        
        logger.debug(f"Parameter devices: {device_info}")
        
        # FINAL FIX: Verify optimizer and model parameter consistency
        if step == 0:
            logger.info("=== FINAL FIX: OPTIMIZER-MODEL PARAMETER CONSISTENCY CHECK ===")
            logger.info(f"Trainable params from model: {[id(p) for p in trainable_params[:3]]}")
            logger.info(f"First param requires_grad: {trainable_params[0].requires_grad if trainable_params else 'N/A'}")
            logger.info(f"First param is_leaf: {trainable_params[0].is_leaf if trainable_params else 'N/A'}")
            logger.info(f"First param grad_fn: {trainable_params[0].grad_fn if trainable_params else 'N/A'}")
            
            # Check if these parameter IDs match what was given to optimizer during setup
            if hasattr(self.selective_finetuner, 'proxy_params'):
                proxy_params = list(self.selective_finetuner.proxy_params.values())
                logger.info(f"Proxy params IDs: {[id(p) for p in proxy_params[:3]]}")
                
                # Check if IDs match
                matches = 0
                for i in range(min(3, len(trainable_params))):
                    model_id = id(trainable_params[i])
                    if i < len(proxy_params):
                        proxy_id = id(proxy_params[i])
                        if model_id == proxy_id:
                            matches += 1
                            logger.info(f"  ✓ Param {i}: IDs match ({model_id})")
                        else:
                            logger.error(f"  ❌ Param {i}: ID MISMATCH - model={model_id}, proxy={proxy_id}")
                            logger.error(f"      This explains why optimizer updates don't work!")
                
                logger.info(f"Parameter ID consistency: {matches}/{min(3, len(trainable_params))} match")
            logger.info("=== END OPTIMIZER DEBUG ===")
        
        self.last_gradient_info = grad_info
        
        # Log every 5 steps to avoid spam
        if step % 5 == 0:
            logger.info(f"[DEBUG] Step {step}: Loss={grad_info['loss']:.6f}, "
                       f"Grads: {grad_info['with_grad']}/{grad_info['with_grad']+grad_info['without_grad']}, "
                       f"Total norm: {grad_info['total_grad_norm']:.6f}, Max: {grad_info['max_grad']:.6f}")
    
    def check_parameter_changes(self, force_log: bool = False):
        """Check if the model parameters (which are now proxy parameters) have changed from initial values."""
        if self.selective_finetuner.proxy_params is None:
            logger.warning("No proxy parameters found for change detection")
            return False
        
        changed_params = 0
        unchanged_params = 0
        total_change_magnitude = 0.0
        
        # Check the actual model parameters since those are what changed during training
        trainable_params = self.selective_finetuner.get_trainable_parameters()
        
        # CRITICAL DEBUG: Check if we have the right number of parameters
        if force_log:
            logger.info(f"[DEBUG] Checking {len(trainable_params)} trainable parameters")
            logger.info(f"[DEBUG] We have {len(self.initial_param_hashes)} initial hashes stored")
            logger.info(f"[DEBUG] First 3 initial hash keys: {list(self.initial_param_hashes.keys())[:3]}")
        
        for i, param in enumerate(trainable_params):
            proxy_name = f"proxy_{i:06d}"
            
            # Convert to float32 first to handle BFloat16
            param_data = param.data.cpu().float()
            param_bytes = param_data.numpy().tobytes()
            current_hash = hashlib.md5(param_bytes).hexdigest()[:8]
            initial_hash = self.initial_param_hashes.get(proxy_name, "unknown")
            
            # CRITICAL DEBUG: Check for hash mismatches
            if force_log and i < 3:
                logger.info(f"[DEBUG] Param {i} ({proxy_name}): initial={initial_hash}, current={current_hash}, same={current_hash == initial_hash}")
                
                # Also check parameter ID consistency
                param_id = id(param)
                logger.info(f"[DEBUG] Param {i} ID: {param_id}")
            
            if current_hash != initial_hash:
                changed_params += 1
                # Calculate magnitude of change for the first few parameters
                if changed_params <= 3:
                    logger.debug(f"Model parameter {i} ({proxy_name}) changed: {initial_hash} -> {current_hash}")
                    
                    # Try a different approach - compare parameter values directly
                    if i < len(trainable_params):
                        # Get current parameter values
                        current_values = param.data.cpu().float()
                        
                        # Try to find the matching initial values from proxy_params
                        # The issue might be in how we match initial vs current parameters
                        matching_initial = None
                        for proxy_key, proxy_param in self.selective_finetuner.proxy_params.items():
                            initial_values = proxy_param.data.cpu().float()
                            if current_values.shape == initial_values.shape:
                                diff = torch.abs(current_values - initial_values).mean().item()
                                if matching_initial is None or diff < matching_initial[1]:
                                    matching_initial = (proxy_key, diff)
                        
                        if matching_initial:
                            total_change_magnitude += matching_initial[1]
                            logger.debug(f"  Best match: {matching_initial[0]}, diff: {matching_initial[1]:.8f}")
            else:
                unchanged_params += 1
                
                # CRITICAL DEBUG: For unchanged params, verify if they should have changed
                if force_log and i < 3:
                    logger.info(f"[DEBUG] Param {i} UNCHANGED - this might be the bug!")
        
        has_changes = changed_params > 0
        
        if force_log or (self.step_count > 0 and self.step_count % 10 == 0):
            logger.info(f"[DEBUG] Parameter changes: {changed_params} changed, {unchanged_params} unchanged")
            if has_changes and total_change_magnitude > 0:
                avg_change = total_change_magnitude / min(changed_params, 3)
                logger.info(f"[DEBUG] Average change magnitude: {avg_change:.8f}")
        
        return has_changes
    
    def check_parameter_changes_direct(self, force_log: bool = False):
        """
        PHASE 4 FIX: More robust parameter change detection using direct value comparison.
        This bypasses potential hash comparison issues.
        """
        if self.selective_finetuner.proxy_params is None:
            logger.warning("No proxy parameters found for change detection")
            return False
        
        changed_params = 0
        unchanged_params = 0
        total_change_magnitude = 0.0
        
        # Get current model parameters
        trainable_params = self.selective_finetuner.get_trainable_parameters()
        
        if force_log:
            logger.info(f"[DIRECT CHECK] Comparing {len(trainable_params)} parameters with their initial values")
        
        # Compare current parameters with initial proxy parameters
        for i, current_param in enumerate(trainable_params):
            proxy_name = f"proxy_{i:06d}"
            
            if proxy_name in self.selective_finetuner.proxy_params:
                initial_proxy = self.selective_finetuner.proxy_params[proxy_name]
                
                # Convert to same device and dtype for comparison
                current_data = current_param.data.cpu().float()
                initial_data = initial_proxy.data.cpu().float()
                
                # Calculate direct difference
                if current_data.shape == initial_data.shape:
                    diff = torch.abs(current_data - initial_data)
                    mean_diff = diff.mean().item()
                    max_diff = diff.max().item()
                    
                    # Consider changed if mean difference > threshold
                    threshold = 1e-8
                    if mean_diff > threshold:
                        changed_params += 1
                        total_change_magnitude += mean_diff
                        
                        if force_log and changed_params <= 3:
                            logger.info(f"[DIRECT CHECK] Parameter {i} CHANGED: mean_diff={mean_diff:.8f}, max_diff={max_diff:.8f}")
                    else:
                        unchanged_params += 1
                        if force_log and unchanged_params <= 3:
                            logger.info(f"[DIRECT CHECK] Parameter {i} unchanged: mean_diff={mean_diff:.8f}")
                else:
                    logger.warning(f"[DIRECT CHECK] Shape mismatch for parameter {i}: {current_data.shape} vs {initial_data.shape}")
                    unchanged_params += 1
            else:
                logger.warning(f"[DIRECT CHECK] No initial proxy found for parameter {i}")
                unchanged_params += 1
        
        has_changes = changed_params > 0
        
        if force_log:
            logger.info(f"[DIRECT CHECK] Results: {changed_params} changed, {unchanged_params} unchanged")
            if has_changes and total_change_magnitude > 0:
                avg_change = total_change_magnitude / changed_params
                logger.info(f"[DIRECT CHECK] Average change magnitude: {avg_change:.8f}")
        
        return has_changes
    
    def pre_sampling_check(self):
        """Comprehensive check before sampling to ensure training had effect."""
        logger.info("=== PRE-SAMPLING DEBUG CHECK ===")
        
        # Check if parameters changed using hash-based method
        has_changes_hash = self.check_parameter_changes(force_log=True)
        
        # PHASE 4 FIX: Also check using direct comparison method
        has_changes_direct = self.check_parameter_changes_direct(force_log=True)
        
        # Check gradient info from last step
        if self.last_gradient_info:
            info = self.last_gradient_info
            logger.info(f"Last gradient info: Loss={info['loss']:.6f}, "
                       f"Grads={info['with_grad']}/{info['with_grad']+info['without_grad']}, "
                       f"Total norm={info['total_grad_norm']:.6f}")
        
        # Apply updates to model
        logger.info("Applying proxy parameter updates to main model...")
        self.selective_finetuner.apply_updates_to_model()
        logger.info("Updates applied successfully")
        
        # Determine if training is working based on both methods
        if has_changes_hash or has_changes_direct:
            logger.info("✅ Parameter changes detected - Training appears to be working")
            if has_changes_hash and not has_changes_direct:
                logger.info("  (Hash method detected changes, direct method did not)")
            elif not has_changes_hash and has_changes_direct:
                logger.info("  (Direct method detected changes, hash method did not)")
            else:
                logger.info("  (Both methods detected changes)")
        else:
            logger.warning("⚠️  NO PARAMETER CHANGES DETECTED BY EITHER METHOD - Training may not be working!")
            
            # PHASE 4 FIX: Additional diagnostics when no changes detected
            logger.warning("=== ADDITIONAL DIAGNOSTICS ===")
            trainable_params = self.selective_finetuner.get_trainable_parameters()
            logger.warning(f"Current trainable parameters: {len(trainable_params)}")
            
            if trainable_params:
                param = trainable_params[0]
                logger.warning(f"First parameter: device={param.device}, requires_grad={param.requires_grad}, is_leaf={param.is_leaf}")
                logger.warning(f"First parameter grad: {param.grad is not None}, grad_fn={param.grad_fn}")
        
        logger.info("=== END DEBUG CHECK ===")
        
        return has_changes_hash or has_changes_direct


def add_selective_training_debug(selective_finetuner) -> SelectiveTrainingDebugger:
    """Create and return a debugger for the selective fine-tuner."""
    return SelectiveTrainingDebugger(selective_finetuner)