#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg") # Use TkAgg backend for rendering plots and env

# --- Boilerplate ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
print = lambda *a, **k: (__import__("builtins").print(*a, **{**k, "flush": True}))

# [NEW] Import RPCClient from training script location or copy definition
# Assuming RPCClient definition is available (e.g., copied or imported)
from multiprocessing.connection import Client

class RPCClient:
    def __init__(self, host, port, authkey="yolo-rpc"):
        self.address = (host, port)
        self.authkey = authkey.encode("utf-8")
        self.conn: Optional[Client] = None
    def __enter__(self):
        try:
            self.conn = Client(self.address, authkey=self.authkey)
            print(f"📡 Connected to RPC server at {self.address}")
        except ConnectionRefusedError:
            print(f"❌ FATAL: Connection to RPC server {self.address} refused. Is it running?")
            raise
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn: self.conn.close()
        print("🔌 Disconnected from RPC server.")
    def reset(self):
        self.conn.send(("reset", None)); _ = self.conn.recv()
    def update_full(self, frame_bgr, steps=0):
        self.conn.send(("update_full", (frame_bgr, steps))); ok, z = self.conn.recv(); return z if ok else None
    def predict_only(self):
        self.conn.send(("predict_only", None)); ok, z = self.conn.recv(); return z if ok else None

# ----------------------------
# 1. Neural Network (Ensure this matches training)
# ----------------------------
class NNPolicy(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        # CRITICAL: Input must be 7
        self.net = nn.Sequential(nn.Linear(7, hidden), nn.ReLU(), nn.Linear(hidden, 4))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def set_weights_vector(m: nn.Module, vec: np.ndarray):
    offset = 0
    with torch.no_grad():
        for p in m.parameters():
            n = p.numel()
            # Ensure device compatibility if loading onto GPU
            device = p.device
            p.data.copy_(torch.from_numpy(vec[offset:offset+n]).view_as(p).to(device))
            offset += n

# ----------------------------
# 2. Model Loading Function
# ----------------------------
def load_model_from_run_dir(run_dir: str, model_type: str = "training_best") -> Optional[NNPolicy]:
    model = NNPolicy()
    weights_path = os.path.join(run_dir, f"best_model_vec_{model_type}.npz")
    state_dict_path = os.path.join(run_dir, f"best_model_{model_type}.pt") # Corrected path

    if os.path.exists(weights_path):
        print(f"✅ Loading weights from NPZ: {weights_path}")
        try:
            weights_data = np.load(weights_path)
            weights_vec = weights_data["weights"]
            set_weights_vector(model, weights_vec)
            return model
        except Exception as e:
            print(f"❌ Failed to load NPZ weights: {e}")
            return None
    elif os.path.exists(state_dict_path):
        print(f"✅ Loading state dict: {state_dict_path}")
        try:
            # Load state dict considering device (CPU default)
            model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu')))
            return model
        except Exception as e:
            print(f"❌ Failed to load PyTorch state dict: {e}")
            return None
    else:
        print(f"❌ Model files ('...vec_{model_type}.npz' or '...{model_type}.pt') not found in {run_dir}")
        return None

# ----------------------------
# 3. Test Function [MODIFIED TO USE RPC]
# ----------------------------
def test_model(
    model: NNPolicy,
    test_seeds: List[int],
    max_steps: int,
    render: bool,
    # [NEW] RPC parameters
    rpc_host: str,
    rpc_port: int,
    rpc_authkey: str
) -> List[float]:
    model.eval()
    rewards_per_seed = []

    # Use 'rgb_array' mode to get frames for RPC, even if rendering window
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    
    # Optional: Separate window for human viewing if --render is used
    viewer_env = None
    if render:
        try:
            viewer_env = gym.make("LunarLander-v3", render_mode="human")
            viewer_env.reset(seed=test_seeds[0]) # Reset once initially
            print("🖥️  Human rendering window enabled.")
        except Exception as e:
            print(f"⚠️ Could not create human rendering window: {e}. Will proceed without.")
            render = False # Disable rendering if it fails

    for i, seed in enumerate(test_seeds):
        print(f"\n--- Testing on Seed {i+1}/{len(test_seeds)} (Seed Value: {seed}) ---")
        total_reward = 0.0
        obs8d, info = env.reset(seed=int(seed))
        
        # Reset viewer env for each seed if rendering
        if viewer_env:
            viewer_env.reset(seed=int(seed))

        last_valid_state7d = np.zeros(7, dtype=np.float32)

        try:
            # Connect to RPC server for this episode
            with RPCClient(rpc_host, rpc_port, rpc_authkey) as rpc:
                rpc.reset()

                for t in range(max_steps):
                    # Get frame from the main env
                    frame_rgb = env.render()
                    if frame_rgb is None: break
                    frame_bgr = frame_rgb[..., ::-1] # RGB -> BGR for YOLO

                    # Get 7D state from RPC server
                    state7d = rpc.update_full(frame_bgr, t)
                    if state7d is None: state7d = rpc.predict_only()
                    if state7d is None: state7d = last_valid_state
                    else: last_valid_state = state7d

                    # --- Action Selection based on 7D state ---
                    s7d = torch.tensor(state7d, dtype=torch.float32)
                    with torch.no_grad():
                        action = int(torch.argmax(model(s7d)).item())
                    # ------------------------------------------
                    
                    # Step the main environment
                    obs8d, reward, done, truncated, info = env.step(action)
                    total_reward += reward

                    # Step the viewer environment if rendering
                    if viewer_env:
                        try:
                            viewer_env.step(action)
                            viewer_env.render()
                        except Exception as render_e:
                            print(f"  (Render step failed: {render_e})")
                            viewer_env.close()
                            viewer_env = None # Stop trying to render

                    if done or truncated:
                        break
        
        except ConnectionRefusedError:
            print("  Skipping seed due to RPC connection failure.")
            rewards_per_seed.append(float('-inf')) # Indicate failure
            continue # Go to next seed
        except Exception as e:
            print(f"  Error during testing on seed {seed}: {e}")
            rewards_per_seed.append(float('-inf')) # Indicate failure
            continue # Go to next seed

        print(f"  Total Reward for Seed {seed}: {total_reward:.2f}")
        rewards_per_seed.append(total_reward)

    env.close()
    if viewer_env:
        viewer_env.close()
        
    return rewards_per_seed

# ----------------------------
# 4. Visualization (No changes needed)
# ----------------------------
def plot_test_results(rewards: List[float], run_dir: str, model_tag: str):
    valid_rewards = [r for r in rewards if r > -np.inf] # Filter out failures
    if not valid_rewards:
        print("No valid rewards recorded to plot.")
        return

    plt.figure(figsize=(12, 6))
    x = np.arange(len(valid_rewards))
    plt.bar(x, valid_rewards, color='skyblue')
    mean_reward = np.mean(valid_rewards)
    std_reward = np.std(valid_rewards)
    plt.xlabel("Test Seed Index (Successful Runs)")
    plt.ylabel("Total Reward")
    plt.title(f"Model Test Results - Avg Reward: {mean_reward:.2f} ± {std_reward:.2f} (Model: {model_tag})")
    plt.xticks(x, [f'Run {i+1}' for i in x], rotation=45, ha='right')
    plt.axhline(y=mean_reward, color='r', linestyle='--', label=f'Average: {mean_reward:.2f}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(run_dir, f"test_results_{model_tag}.png")
    plt.savefig(output_path, dpi=150)
    print(f"📈 Test results plot saved to {output_path}")
    plt.show()

# ----------------------------
# 5. Main Execution
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Test trained GA model on LunarLander-v3 using RPC state.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the GA training run directory.")
    parser.add_argument("--model-type", type=str, default="training_best", choices=["training_best", "full_eval"], help="Which model version to load.")
    
    # [NEW] RPC arguments for testing
    parser.add_argument("--rpc-host", type=str, default="127.0.0.1", help="RPC server host.")
    parser.add_argument("--rpc-port", type=int, default=6001, help="RPC server port.")
    parser.add_argument("--rpc-authkey", type=str, default="yolo-rpc", help="RPC authentication key.")

    parser.add_argument("--num-tests", type=int, default=10, help="Number of different seeds to test.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per episode.")
    parser.add_argument("--start-seed", type=int, default=1000, help="Starting seed for generating test seeds.")
    parser.add_argument("--render", action="store_true", help="Render the environment window during testing.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.start_seed)
    test_seeds = rng.integers(low=0, high=2**31-1, size=args.num_tests, dtype=np.int32).tolist()
    print(f"Generated {args.num_tests} test seeds: {test_seeds}")

    model = load_model_from_run_dir(args.run_dir, args.model_type)
    if model is None: return

    all_rewards = test_model(
        model, test_seeds, args.max_steps, args.render,
        args.rpc_host, args.rpc_port, args.rpc_authkey # Pass RPC args
    )

    valid_rewards = [r for r in all_rewards if r > -np.inf]
    if valid_rewards:
        mean_reward = np.mean(valid_rewards)
        std_reward = np.std(valid_rewards)
        max_reward = np.max(valid_rewards)
        min_reward = np.min(valid_rewards)
        print(f"\n--- Test Summary ({len(valid_rewards)} successful runs out of {args.num_tests}) ---")
        print(f"Individual rewards: {[f'{r:.2f}' for r in valid_rewards]}")
        print(f"Average reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Max reward: {max_reward:.2f}")
        print(f"Min reward: {min_reward:.2f}")
        plot_test_results(valid_rewards, args.run_dir, args.model_type)
    else:
        print("\n--- Test Summary ---")
        print("No successful runs completed (check RPC server and model).")

if __name__ == "__main__":
    main()