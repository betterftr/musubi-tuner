"""
Minimal debugging utilities for selective fine-tuning training
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import hashlib
from .selective_finetune import TURN_ON_DEBUGS

logger = logging.getLogger(__name__)

# Make logger conditional based on TURN_ON_DEBUGS flag
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


class SelectiveTrainingDebugger:
    """Minimal debugger for tracking gradient flow and weight changes in selective fine-tuning."""
    
    def __init__(self, selective_finetuner):
        self.selective_finetuner = selective_finetuner
        self.initial_param_hashes = {}
        self.initial_param_values = {}
        self.step_count = 0
        self.last_gradient_info = {}
        
        # Store initial parameter values for change detection
        self._store_initial_values()
    
    def _store_initial_values(self):
        """Store both hashes and copies of initial parameter values for robust change detection."""
        if self.selective_finetuner.proxy_params is not None and self.selective_finetuner.param_metadata is not None:
            # Get the actual parameters from the model (which should now be the proxy parameters)
            model_params = dict(self.selective_finetuner.model.named_parameters())
            
            for proxy_name in self.selective_finetuner.param_metadata:
                original_name, _, _ = self.selective_finetuner.param_metadata[proxy_name]
                
                if original_name in model_params:
                    # Use the actual parameter from the model (which should be the proxy)
                    model_param = model_params[original_name]
                    
                    # Convert to float32 first to handle BFloat16
                    param_data = model_param.data.cpu().float()
                    
                    # Store hash
                    param_bytes = param_data.numpy().tobytes()
                    param_hash = hashlib.md5(param_bytes).hexdigest()[:8]
                    self.initial_param_hashes[original_name] = param_hash
                    
                    # Store a DEEP COPY of the initial values (critical for proper comparison)
                    self.initial_param_values[original_name] = param_data.clone().detach()
                    
                    logger.debug(f"Initial values stored for {original_name}: hash={param_hash}, shape={param_data.shape}")
            
            logger.debug(f"Stored initial values for {len(self.initial_param_hashes)} parameters")
    
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
        
        # Get the actual trainable parameters from the model
        model_params = dict(self.selective_finetuner.model.named_parameters())
        selected_params = []
        
        # Get the parameters that should be trainable based on our metadata
        for proxy_name in self.selective_finetuner.param_metadata:
            original_name, _, _ = self.selective_finetuner.param_metadata[proxy_name]
            if original_name in model_params:
                param = model_params[original_name]
                selected_params.append((original_name, param))
        
        logger.debug(f"Checking gradients on {len(selected_params)} selective fine-tuning parameters")
        
        # Check device placement and gradient status
        device_info = {}
        for i, (param_name, param) in enumerate(selected_params):
            device_str = str(param.device)
            device_info[device_str] = device_info.get(device_str, 0) + 1
            
            if param.grad is not None:
                grad_info['with_grad'] += 1
                grad_norm = param.grad.data.norm().item()
                grad_info['total_grad_norm'] += grad_norm
                grad_info['max_grad'] = max(grad_info['max_grad'], grad_norm)
                if i < 3:  # Log first few for debugging
                    logger.debug(f"Param {param_name}: grad norm = {grad_norm:.8f}, device={param.device}")
            else:
                grad_info['without_grad'] += 1
                if i < 3:  # Log first few for debugging
                    logger.debug(f"Param {param_name}: NO GRADIENT, device={param.device}, requires_grad={param.requires_grad}")
        
        logger.debug(f"Parameter devices: {device_info}")
        
        # Parameter consistency check on first step
        if step == 0:
            logger.info("=== PARAMETER CONSISTENCY CHECK ===")
            logger.info(f"Found {len(selected_params)} selective fine-tuning parameters in model")
            logger.info(f"Parameters with requires_grad=True: {sum(1 for _, p in selected_params if p.requires_grad)}")
            
            # Check if the model parameters match our proxy parameters
            proxy_param_ids = {id(p) for p in self.selective_finetuner.proxy_params.values()}
            model_param_ids = {id(p) for _, p in selected_params}
            
            matching_ids = proxy_param_ids.intersection(model_param_ids)
            logger.info(f"Parameter ID matches: {len(matching_ids)}/{len(proxy_param_ids)} proxy params found in model")
            
            if len(matching_ids) != len(proxy_param_ids):
                logger.warning("Some proxy parameters are not in the model - this may cause training issues")
            
            logger.info("=== END CONSISTENCY CHECK ===")
        
        self.last_gradient_info = grad_info
        
        # Log every 5 steps to avoid spam
        if step % 5 == 0:
            logger.info(f"[DEBUG] Step {step}: Loss={grad_info['loss']:.6f}, "
                       f"Grads: {grad_info['with_grad']}/{grad_info['with_grad']+grad_info['without_grad']}, "
                       f"Total norm: {grad_info['total_grad_norm']:.6f}, Max: {grad_info['max_grad']:.6f}")
    
    def check_parameter_changes(self, force_log: bool = False):
        """Check if the model parameters have changed from initial values using hash comparison."""
        if self.selective_finetuner.proxy_params is None or self.selective_finetuner.param_metadata is None:
            logger.warning("No proxy parameters or metadata found for change detection")
            return False
        
        if not self.initial_param_hashes:
            logger.warning("[HASH CHECK] No initial hashes stored - cannot perform comparison")
            return False
        
        changed_params = 0
        unchanged_params = 0
        
        if force_log:
            logger.info(f"[DEBUG] Checking {len(self.initial_param_hashes)} trainable parameters")
            logger.info(f"[DEBUG] We have {len(self.initial_param_hashes)} initial hashes stored")
            logger.info(f"[DEBUG] First 3 initial hash keys: {list(self.initial_param_hashes.keys())[:3]}")
        
        # Get current model parameters
        model_params = dict(self.selective_finetuner.model.named_parameters())
        
        # Check each parameter that we have initial hashes for
        for original_name, initial_hash in self.initial_param_hashes.items():
            if original_name not in model_params:
                logger.warning(f"[HASH CHECK] Parameter {original_name} not found in current model")
                continue
                
            current_param = model_params[original_name]
            
            # Convert to float32 first to handle BFloat16
            param_data = current_param.data.cpu().float()
            param_bytes = param_data.numpy().tobytes()
            current_hash = hashlib.md5(param_bytes).hexdigest()[:8]
            
            # Compare hashes
            if current_hash != initial_hash:
                changed_params += 1
                if force_log and changed_params <= 3:
                    logger.info(f"[DEBUG] Param {original_name}: initial={initial_hash}, current={current_hash}, CHANGED")
                    param_id = id(current_param)
                    logger.info(f"[DEBUG] Param {original_name} ID: {param_id}")
            else:
                unchanged_params += 1
                
                # For unchanged params, check why they might not be changing
                if force_log and unchanged_params <= 3:
                    logger.info(f"[DEBUG] Param {original_name} UNCHANGED")
                    param_id = id(current_param)
                    logger.info(f"[DEBUG] Param {original_name} ID: {param_id}")
                    
                    # Check if this is due to the parameter not being trained
                    if not current_param.requires_grad:
                        logger.info(f"  -> Parameter has requires_grad=False")
                    elif current_param.grad is None:
                        logger.info(f"  -> Parameter has no gradient")
                    else:
                        logger.info(f"  -> Parameter has gradient but values unchanged (may indicate issue)")
        
        has_changes = changed_params > 0
        
        if force_log:
            logger.info(f"[DEBUG] Parameter changes: {changed_params} changed, {unchanged_params} unchanged")
        
        return has_changes
    
    def check_parameter_changes_direct(self, force_log: bool = False):
        """
        Robust parameter change detection using direct value comparison against stored initial values.
        Enhanced with block swap awareness.
        """
        if self.selective_finetuner.proxy_params is None or self.selective_finetuner.param_metadata is None:
            logger.warning("No proxy parameters or metadata found for change detection")
            return False
        
        if not self.initial_param_values:
            logger.warning("[DIRECT CHECK] No initial values stored - cannot perform comparison")
            return False
        
        changed_params = 0
        unchanged_params = 0
        total_change_magnitude = 0.0
        changed_swapped = 0
        changed_non_swapped = 0
        unchanged_swapped = 0
        unchanged_non_swapped = 0
        
        # Get block swapping info
        blocks_to_swap = getattr(self.selective_finetuner.model, 'blocks_to_swap', 0)
        swapped_block_start = 40 - blocks_to_swap if blocks_to_swap > 0 else 40
        
        if force_log:
            logger.info(f"[DIRECT CHECK] Comparing {len(self.initial_param_values)} parameters with stored initial values")
        
        # Get current model parameters
        model_params = dict(self.selective_finetuner.model.named_parameters())
        
        # Compare current parameters with stored initial values
        for original_name, initial_data in self.initial_param_values.items():
            if original_name not in model_params:
                logger.warning(f"[DIRECT CHECK] Parameter {original_name} not found in current model")
                continue
                
            current_param = model_params[original_name]
            
            # Determine if this is a swapped block parameter
            is_swapped_block = False
            if 'blocks.' in original_name and blocks_to_swap > 0:
                try:
                    block_parts = original_name.split('.')
                    if len(block_parts) >= 2 and block_parts[0] == 'blocks':
                        block_num = int(block_parts[1])
                        is_swapped_block = block_num >= swapped_block_start
                except (ValueError, IndexError):
                    pass
            
            # Convert current to same format for comparison (CPU float32)
            # Handle potential device mismatches from block swapping
            try:
                current_data = current_param.data.cpu().float()
            except Exception as e:
                logger.warning(f"[DIRECT CHECK] Failed to move {original_name} to CPU: {e}")
                unchanged_params += 1
                if is_swapped_block:
                    unchanged_swapped += 1
                else:
                    unchanged_non_swapped += 1
                continue
            
            # Verify we have the stored initial data
            if initial_data is None:
                logger.warning(f"[DIRECT CHECK] Initial data is None for {original_name}")
                unchanged_params += 1
                if is_swapped_block:
                    unchanged_swapped += 1
                else:
                    unchanged_non_swapped += 1
                continue
            
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
                    if is_swapped_block:
                        changed_swapped += 1
                    else:
                        changed_non_swapped += 1
                    
                    if force_log and changed_params <= 3:
                        block_type = "[SWAPPED]" if is_swapped_block else "[NON-SWAPPED]"
                        logger.info(f"[DIRECT CHECK] {block_type} Parameter {original_name} CHANGED: mean_diff={mean_diff:.8f}, max_diff={max_diff:.8f}")
                        logger.info(f"  Current mean: {current_data.mean():.8f}, Initial mean: {initial_data.mean():.8f}")
                else:
                    unchanged_params += 1
                    if is_swapped_block:
                        unchanged_swapped += 1
                    else:
                        unchanged_non_swapped += 1
                    if force_log and unchanged_params <= 3:
                        block_type = "[SWAPPED]" if is_swapped_block else "[NON-SWAPPED]"
                        logger.info(f"[DIRECT CHECK] {block_type} Parameter {original_name} unchanged: mean_diff={mean_diff:.8f}")
            else:
                logger.warning(f"[DIRECT CHECK] Shape mismatch for {original_name}: current={current_data.shape} vs initial={initial_data.shape}")
                unchanged_params += 1
                if is_swapped_block:
                    unchanged_swapped += 1
                else:
                    unchanged_non_swapped += 1
        
        has_changes = changed_params > 0
        
        if force_log:
            logger.info(f"[DIRECT CHECK] Results: {changed_params} changed, {unchanged_params} unchanged")
            if blocks_to_swap > 0:
                logger.info(f"  Non-swapped blocks: {changed_non_swapped} changed, {unchanged_non_swapped} unchanged")
                logger.info(f"  Swapped blocks: {changed_swapped} changed, {unchanged_swapped} unchanged")
            if has_changes and total_change_magnitude > 0:
                avg_change = total_change_magnitude / changed_params
                logger.info(f"[DIRECT CHECK] Average change magnitude: {avg_change:.8f}")
        
        return has_changes
    
    def pre_sampling_check(self):
        """Comprehensive check before sampling to ensure training had effect."""
        logger.info("=== PRE-SAMPLING DEBUG CHECK ===")
        
        # BLOCK SWAP DEBUG: Show parameter distribution by block
        self._log_block_swap_parameter_distribution()
        
        # Force device synchronization before parameter checks
        self._force_device_synchronization()
        
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
        
        # Parameters should be directly in the model and changed by training
        logger.info("Checking if parameters in model have changed from training...")
        
        # Also verify we're checking the right parameters
        model_params = dict(self.selective_finetuner.model.named_parameters())
        trainable_count = 0
        for proxy_name in self.selective_finetuner.param_metadata:
            original_name, _, _ = self.selective_finetuner.param_metadata[proxy_name]
            if original_name in model_params:
                param = model_params[original_name]
                if param.requires_grad:
                    trainable_count += 1
        logger.info(f"Verified {trainable_count} trainable parameters in model for change detection")
        
        # Check if gradients are actually flowing on the parameters we're monitoring
        model_params = dict(self.selective_finetuner.model.named_parameters())
        params_with_grads = 0
        total_monitored_params = 0
        swapped_params_with_grads = 0
        non_swapped_params_with_grads = 0
        
        # Get block swapping info for categorization
        blocks_to_swap = getattr(self.selective_finetuner.model, 'blocks_to_swap', 0)
        swapped_block_start = 40 - blocks_to_swap if blocks_to_swap > 0 else 40
        
        for proxy_name in self.selective_finetuner.param_metadata:
            original_name, _, _ = self.selective_finetuner.param_metadata[proxy_name]
            if original_name in model_params:
                param = model_params[original_name]
                total_monitored_params += 1
                
                # Categorize by block type
                is_swapped_block = False
                if 'blocks.' in original_name:
                    try:
                        block_parts = original_name.split('.')
                        if len(block_parts) >= 2 and block_parts[0] == 'blocks':
                            block_num = int(block_parts[1])
                            is_swapped_block = block_num >= swapped_block_start
                    except (ValueError, IndexError):
                        pass
                
                if param.grad is not None:
                    params_with_grads += 1
                    if is_swapped_block:
                        swapped_params_with_grads += 1
                    else:
                        non_swapped_params_with_grads += 1
        
        # Determine if training is working
        training_working = False
        if params_with_grads > 0:
            logger.info(f"✅ Gradients detected on {params_with_grads}/{total_monitored_params} monitored parameters")
            if blocks_to_swap > 0:
                logger.info(f"  Non-swapped blocks: {non_swapped_params_with_grads} with gradients")
                logger.info(f"  Swapped blocks: {swapped_params_with_grads} with gradients")
            training_working = True
        
        if has_changes_hash or has_changes_direct:
            logger.info("✅ Parameter changes detected - Training appears to be working")
            if has_changes_hash and not has_changes_direct:
                logger.info("  (Hash method detected changes, direct method did not)")
            elif not has_changes_hash and has_changes_direct:
                logger.info("  (Direct method detected changes, hash method did not)")
            else:
                logger.info("  (Both methods detected changes)")
            training_working = True
        
        # FINAL VERIFICATION: Check if ALL selected parameters are being trained
        final_verification = self._perform_final_training_verification(has_changes_hash, has_changes_direct, training_working)
        
        if not final_verification:
            logger.warning("⚠️  PARAMETER COVERAGE ISSUE - Some parameters may not be training!")
            logger.warning("This could indicate block swapping compatibility issues")
        else:
            logger.info("✅ Parameter coverage verification PASSED - All selected parameters training")
        
        logger.info("=== END DEBUG CHECK ===")
        
        return final_verification
    
    def _perform_final_training_verification(self, has_changes_hash: bool, has_changes_direct: bool, training_working: bool) -> bool:
        """Verify training effectiveness based on parameter changes and gradient history."""
        logger.info("[PARAMETER COVERAGE] Analyzing training effectiveness...")
        
        # Check if we have evidence that training is working
        evidence_count = 0
        total_evidence = 4
        
        # Evidence 1: Parameter changes detected by hash method (most reliable)
        if has_changes_hash:
            evidence_count += 1
            logger.info("✓ [EVIDENCE] Parameter changes detected by hash comparison")
        else:
            logger.warning("✗ [EVIDENCE] No parameter changes detected by hash comparison")
        
        # Evidence 2: Parameter changes detected by direct method 
        if has_changes_direct:
            evidence_count += 1
            logger.info("✓ [EVIDENCE] Parameter changes detected by direct comparison")
        else:
            logger.info("→ [EVIDENCE] No parameter changes by direct comparison (may be timing issue)")
        
        # Evidence 3: Gradient flow in last training step
        if self.last_gradient_info and self.last_gradient_info.get('with_grad', 0) > 0:
            evidence_count += 1
            grad_count = self.last_gradient_info['with_grad']
            total_count = self.last_gradient_info['with_grad'] + self.last_gradient_info['without_grad']
            logger.info(f"✓ [EVIDENCE] Gradients detected in last step: {grad_count}/{total_count}")
        else:
            logger.warning("✗ [EVIDENCE] No gradient information from last training step")
        
        # Evidence 4: Reasonable loss values
        if self.last_gradient_info and 'loss' in self.last_gradient_info:
            loss = self.last_gradient_info['loss']
            if 0.001 < loss < 10.0:  # Reasonable loss range for fine-tuning
                evidence_count += 1
                logger.info(f"✓ [EVIDENCE] Loss in reasonable range: {loss:.6f}")
            else:
                logger.warning(f"✗ [EVIDENCE] Loss outside expected range: {loss:.6f}")
        else:
            logger.warning("✗ [EVIDENCE] No loss information available")
        
        # Check parameter distribution
        blocks_to_swap = getattr(self.selective_finetuner.model, 'blocks_to_swap', 0)
        swapped_block_start = 40 - blocks_to_swap if blocks_to_swap > 0 else 40
        
        # Count parameters by block type based on changes detected
        non_swapped_changed = 0
        swapped_changed = 0
        
        # Use the hash comparison results to categorize changed parameters
        for proxy_name in self.selective_finetuner.param_metadata:
            original_name, _, _ = self.selective_finetuner.param_metadata[proxy_name]
            
            # Check if this parameter was in our hash comparison and changed
            if original_name in self.initial_param_hashes:
                model_params = dict(self.selective_finetuner.model.named_parameters())
                if original_name in model_params:
                    param = model_params[original_name]
                    param_data = param.data.cpu().float()
                    param_bytes = param_data.numpy().tobytes()
                    current_hash = hashlib.md5(param_bytes).hexdigest()[:8]
                    initial_hash = self.initial_param_hashes[original_name]
                    
                    if current_hash != initial_hash:  # Parameter changed
                        # Determine if this is a swapped block
                        is_swapped_block = False
                        if 'blocks.' in original_name and blocks_to_swap > 0:
                            try:
                                block_parts = original_name.split('.')
                                if len(block_parts) >= 2 and block_parts[0] == 'blocks':
                                    block_num = int(block_parts[1])
                                    is_swapped_block = block_num >= swapped_block_start
                            except (ValueError, IndexError):
                                pass
                        
                        if is_swapped_block:
                            swapped_changed += 1
                        else:
                            non_swapped_changed += 1
        
        total_changed = non_swapped_changed + swapped_changed
        logger.info(f"[PARAMETER COVERAGE] Changed parameters by block type:")
        logger.info(f"  Non-swapped blocks (0-{swapped_block_start-1}): {non_swapped_changed} changed")
        logger.info(f"  Swapped blocks ({swapped_block_start}-39): {swapped_changed} changed")
        logger.info(f"  Total changed: {total_changed}/44 parameters ({100*total_changed/44:.1f}%)")
        
        # Determine overall training effectiveness
        evidence_ratio = evidence_count / total_evidence
        logger.info(f"[VERIFICATION] Training evidence: {evidence_count}/{total_evidence} ({evidence_ratio:.2%})")
        
        # Training is working if we have strong evidence AND parameters are changing
        training_effective = evidence_ratio >= 0.75 and total_changed > 0
        
        if training_effective:
            logger.info("✅ [VERIFICATION] Training is working effectively")
            if swapped_changed > 0 and non_swapped_changed > 0:
                logger.info("  Both swapped and non-swapped block parameters are training")
            elif swapped_changed == 0 and non_swapped_changed > 0:
                logger.info("ℹ️  Only non-swapped parameters showing changes (may be block swap timing)")
            elif swapped_changed > 0 and non_swapped_changed == 0:
                logger.info("ℹ️  Only swapped parameters showing changes (unexpected but working)")
            return True
        else:
            logger.error(f"❌ [VERIFICATION] Training effectiveness insufficient: {evidence_count}/{total_evidence} evidence, {total_changed} changed params")
            return False
    
    def _log_block_swap_parameter_distribution(self):
        """Log which parameters are in swapped vs non-swapped blocks for debugging."""
        if self.selective_finetuner.proxy_params is None or self.selective_finetuner.param_metadata is None:
            return
        
        # Get block swapping info from the model if available
        model = self.selective_finetuner.model
        blocks_to_swap = getattr(model, 'blocks_to_swap', None)
        
        if blocks_to_swap is None or blocks_to_swap == 0:
            logger.info("[BLOCK SWAP DEBUG] Block swapping not enabled")
            return
        
        # Calculate swapped block range
        total_blocks = 40  # Default for Wan DiT
        for name, _ in model.named_parameters():
            if 'blocks.' in name:
                try:
                    block_parts = name.split('.')
                    if len(block_parts) >= 2 and block_parts[0] == 'blocks':
                        block_num = int(block_parts[1])
                        total_blocks = max(total_blocks, block_num + 1)
                except (ValueError, IndexError):
                    continue
        
        swapped_block_start = total_blocks - blocks_to_swap
        logger.info(f"[BLOCK SWAP DEBUG] Total blocks: {total_blocks}, Swapped: {blocks_to_swap} (blocks {swapped_block_start}-{total_blocks-1})")
        
        # Categorize parameters by block type
        non_swapped_params = []
        swapped_params = []
        non_block_params = []
        
        for proxy_name in self.selective_finetuner.param_metadata:
            original_name, _, _ = self.selective_finetuner.param_metadata[proxy_name]
            
            if 'blocks.' in original_name:
                try:
                    block_parts = original_name.split('.')
                    if len(block_parts) >= 2 and block_parts[0] == 'blocks':
                        block_num = int(block_parts[1])
                        if block_num >= swapped_block_start:
                            swapped_params.append((original_name, block_num))
                        else:
                            non_swapped_params.append((original_name, block_num))
                    else:
                        non_block_params.append(original_name)
                except (ValueError, IndexError):
                    non_block_params.append(original_name)
            else:
                non_block_params.append(original_name)
        
        logger.info(f"[BLOCK SWAP DEBUG] Parameter distribution:")
        logger.info(f"  Non-swapped blocks (0-{swapped_block_start-1}): {len(non_swapped_params)} params")
        logger.info(f"  Swapped blocks ({swapped_block_start}-{total_blocks-1}): {len(swapped_params)} params")
        logger.info(f"  Non-block parameters: {len(non_block_params)} params")
        
        if len(non_swapped_params) > 0:
            logger.info(f"  First few non-swapped: {[name.split('.')[:3] for name, _ in non_swapped_params[:3]]}")
        if len(swapped_params) > 0:
            logger.info(f"  First few swapped: {[name.split('.')[:3] for name, _ in swapped_params[:3]]}")
    
    def _force_device_synchronization(self):
        """Force device synchronization to ensure parameters are in stable state for checking."""
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.debug("[DEVICE SYNC] Device synchronization complete")
        except Exception as e:
            logger.debug(f"[DEVICE SYNC] Failed to synchronize: {e}")


def add_selective_training_debug(selective_finetuner) -> SelectiveTrainingDebugger:
    """Create and return a debugger for the selective fine-tuner."""
    debugger = SelectiveTrainingDebugger(selective_finetuner)
    
    # Add a method to the selective finetuner to update debug state when parameters change
    def _update_debug_on_param_change():
        """Update debug state when parameters are moved or modified."""
        if hasattr(debugger, '_store_initial_values'):
            # Clear old values and re-store current state
            debugger.initial_param_values.clear()
            debugger.initial_param_hashes.clear()
            debugger._store_initial_values()
            logger.debug("Updated debug initial values after parameter changes")
    
    # Attach the update method to the finetuner
    selective_finetuner._update_debug_state = _update_debug_on_param_change
    
    return debugger