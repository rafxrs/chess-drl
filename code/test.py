import os
import time
import torch
import chess
import psutil
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import argparse
import config

from datetime import datetime
from matplotlib.ticker import MaxNLocator
from agent import Agent
from env import Chess_Env
from mcts import MCTS
from modelbuilder import RLModelBuilder
from train import move_to_index, batchify, generate_selfplay_data, generate_game


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('chess-drl-test')

class TestSuite:
    def __init__(self, model_path=None):
        """
        Initialize the test suite with an optional model path.
        If not provided, will create an initial model.
        """
        self.model_path = model_path
        self.results = {}
        self.test_model = None
        
        # Create model directories if they don't exist
        os.makedirs(config.MODEL_FOLDER, exist_ok=True)
        os.makedirs("test_results", exist_ok=True)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.USE_GPU else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize model if needed
        if not model_path or not os.path.exists(model_path):
            self.model_path = f"{config.MODEL_FOLDER}/test_model.pt"
            logger.info(f"Creating new model at {self.model_path}")
            self.test_model = RLModelBuilder(
                config.INPUT_SHAPE, config.OUTPUT_SHAPE[0], config.OUTPUT_SHAPE[1]
            ).build_model()
            torch.save(self.test_model.state_dict(), self.model_path)
        else:
            self.test_model = RLModelBuilder(
                config.INPUT_SHAPE, config.OUTPUT_SHAPE[0], config.OUTPUT_SHAPE[1]
            ).build_model(self.model_path)
        
        # Move model to appropriate device
        self.test_model = self.test_model.to(self.device)
    
    def run_all_tests(self):
        """Run all tests and return consolidated results."""
        logger.info("Running all tests...")
        start_time = time.time()
        
        # Core component tests
        self.test_environment()
        self.test_agent_initialization()
        self.test_mcts()
        self.test_model_architecture()
        self.test_move_indexing()
        self.test_data_generation()
        
        # Resource tests
        self.test_memory_usage()
        self.test_cpu_usage()
        if torch.cuda.is_available():
            self.test_gpu_usage()
        
        # Performance tests
        self.test_simulation_speed()
        self.test_batch_processing()
        
        # Generate report
        elapsed = time.time() - start_time
        self.results['total_time'] = elapsed
        logger.info(f"All tests completed in {elapsed:.2f} seconds")
        self.generate_test_report()
        
        return self.results
    
    def test_environment(self):
        """Test chess environment functionality."""
        logger.info("Testing chess environment...")
        try:
            # Create environment
            env = Chess_Env()
            
            # Test reset
            board = env.reset()
            assert board.fen() == chess.STARTING_FEN, "Reset failed to set starting position"
            
            # Test making moves
            e4 = chess.Move.from_uci("e2e4")
            env.push(e4)
            assert env.board.piece_at(chess.E4) is not None, "Move not applied correctly"
            
            # Test state_to_tensor
            tensor = env.state_to_tensor()
            assert tensor.shape[1:] == (config.amount_of_input_planes, 8, 8), f"Incorrect tensor shape: {tensor.shape}"
            
            # Test game outcome detection
            env.reset()
            # Scholar's mate
            for move_uci in ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]:
                env.push(chess.Move.from_uci(move_uci))
            
            assert env.is_game_over(), "Game should be over after Scholar's mate"
            assert env.get_result() == "1-0", "White should win after Scholar's mate"
            
            self.results['environment_test'] = 'PASS'
            logger.info("✅ Environment test passed")
        
        except Exception as e:
            self.results['environment_test'] = f'FAIL: {str(e)}'
            logger.error(f"❌ Environment test failed: {e}")
            raise
    
    def test_agent_initialization(self):
        """Test agent initialization and basic functionality."""
        logger.info("Testing agent initialization...")
        try:
            # Initialize agent with test model
            agent = Agent(model_path=self.model_path)
            
            # Check model loaded properly
            assert agent.model is not None, "Model not initialized"
            
            # Test run_simulations doesn't crash
            agent.state = chess.STARTING_FEN
            agent.run_simulations(n=10)  # Smaller number for faster test
            
            # Check if agent can produce a valid move
            env = Chess_Env()
            move = agent.get_move(env)
            assert isinstance(move, chess.Move), f"Invalid move type: {type(move)}"
            assert move in env.board.legal_moves, "Illegal move generated"
            
            self.results['agent_test'] = 'PASS'
            logger.info("✅ Agent initialization test passed")
        
        except Exception as e:
            self.results['agent_test'] = f'FAIL: {str(e)}'
            logger.error(f"❌ Agent initialization test failed: {e}")
            raise
    
    def test_mcts(self):
        """Test Monte Carlo Tree Search functionality."""
        logger.info("Testing MCTS...")
        try:
            # Create MCTS instance
            agent = Agent(model_path=self.model_path)
            mcts = MCTS(agent, config.__dict__)
            
            # Test simulation
            agent.state = chess.STARTING_FEN
            mcts.run_simulation(agent.model, n=5)  # Small number for testing
            
            # Check if root node was expanded
            assert len(mcts.root.children) > 0, "MCTS root node not expanded"
            
            # Get move probabilities
            actions, probs = mcts.get_move_probs()
            
            # Check if valid moves and probabilities
            assert len(actions) > 0, "No actions returned from MCTS"
            assert len(probs) == len(actions), "Actions and probabilities length mismatch"
            assert abs(sum(probs) - 1.0) < 1e-5, f"Probabilities don't sum to 1: {sum(probs)}"
            
            # Check for NaN values
            assert not np.isnan(probs).any(), "NaN values in move probabilities"
            
            self.results['mcts_test'] = 'PASS'
            logger.info("✅ MCTS test passed")
        
        except Exception as e:
            self.results['mcts_test'] = f'FAIL: {str(e)}'
            logger.error(f"❌ MCTS test failed: {e}")
            raise
    
    def test_model_architecture(self):
        """Test neural network architecture and forward pass."""
        logger.info("Testing neural network model...")
        try:
            # Create fake input batch
            batch_size = 2
            input_channels = config.amount_of_input_planes
            fake_input = torch.randn(batch_size, input_channels, 8, 8).to(self.device)
            
            # Test forward pass
            self.test_model.eval()
            with torch.no_grad():
                policy_output, value_output = self.test_model(fake_input)
            
            # Check output shapes
            assert policy_output.shape == (batch_size, config.OUTPUT_SHAPE[0]), f"Policy output shape wrong: {policy_output.shape}"
            assert value_output.shape == (batch_size, config.OUTPUT_SHAPE[1]), f"Value output shape wrong: {value_output.shape}"
            
            # Check output ranges
            assert policy_output.min() >= 0, "Policy values should be positive (sigmoid output)"
            assert policy_output.max() <= 1, "Policy values should be <= 1 (sigmoid output)"
            assert value_output.min() >= -1, "Value should be >= -1 (tanh output)"
            assert value_output.max() <= 1, "Value should be <= 1 (tanh output)"
            
            # Gradient test
            self.test_model.train()
            policy_output, value_output = self.test_model(fake_input)
            loss = policy_output.mean() + value_output.mean()
            loss.backward()
            
            # Check if gradients are computed
            for name, param in self.test_model.named_parameters():
                assert param.grad is not None, f"No gradient for {name}"
            
            self.results['model_test'] = 'PASS'
            logger.info("✅ Model architecture test passed")
        
        except Exception as e:
            self.results['model_test'] = f'FAIL: {str(e)}'
            logger.error(f"❌ Model architecture test failed: {e}")
            raise
    
    def test_move_indexing(self):
        """Test move_to_index function for correct policy vector mapping."""
        logger.info("Testing move encoding...")
        try:
            # Test various move types
            board = chess.Board()
            
            # Queen-like moves
            e4 = chess.Move.from_uci("e2e4")
            e4_idx = move_to_index(e4)
            
            # Knight move
            knight = chess.Move.from_uci("b1c3")
            knight_idx = move_to_index(knight)
            
            # Test castling
            castle_board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
            castle_move = chess.Move.from_uci("e1g1")  # King-side castling
            castle_idx = move_to_index(castle_move)
            
            # Test promotion
            prom_board = chess.Board("8/P7/8/8/8/8/8/8 w - - 0 1")
            prom_move = chess.Move.from_uci("a7a8q")  # Queen promotion
            prom_idx = move_to_index(prom_move)
            
            # Check for index ranges
            moves = [e4, knight, castle_move, prom_move]
            indices = [e4_idx, knight_idx, castle_idx, prom_idx]
            
            for move, idx in zip(moves, indices):
                assert 0 <= idx < config.OUTPUT_SHAPE[0], f"Index out of range for {move}: {idx}"
            
            # Check for uniqueness
            assert len(set(indices)) == len(indices), "Duplicate move indices found"
            
            self.results['move_indexing_test'] = 'PASS'
            logger.info("✅ Move indexing test passed")
        
        except Exception as e:
            self.results['move_indexing_test'] = f'FAIL: {str(e)}'
            logger.error(f"❌ Move indexing test failed: {e}")
            raise
    
    def test_data_generation(self):
        """Test self-play data generation."""
        logger.info("Testing data generation...")
        try:
            # Test single game generation
            agent = Agent(model_path=self.model_path)
            
            # Small config for testing
            original_sim_count = config.SIMULATIONS_PER_MOVE
            original_max_moves = config.MAX_GAME_MOVES
            
            # Override for faster testing
            config.SIMULATIONS_PER_MOVE = 5
            config.MAX_GAME_MOVES = 20
            
            # Generate a test game
            states, policies, values = generate_game(self.model_path, game_id=0)
            
            # Check outputs
            assert len(states) > 0, "No states generated"
            assert len(policies) == len(states), "Policies and states length mismatch"
            assert len(values) == len(states), "Values and states length mismatch"
            
            # Check policy vectors
            for policy in policies:
                assert policy.shape == (config.OUTPUT_SHAPE[0],), f"Wrong policy shape: {policy.shape}"
                assert abs(np.sum(policy) - 1.0) < 0.1, f"Policy doesn't sum close to 1: {np.sum(policy)}"
            
            # Check value targets
            for value in values:
                assert -1 <= value <= 1, f"Value out of range: {value}"
            
            # Restore original config
            config.SIMULATIONS_PER_MOVE = original_sim_count
            config.MAX_GAME_MOVES = original_max_moves
            
            self.results['data_generation_test'] = 'PASS'
            logger.info("✅ Data generation test passed")
        
        except Exception as e:
            # Restore original config even if test fails
            config.SIMULATIONS_PER_MOVE = original_sim_count
            config.MAX_GAME_MOVES = original_max_moves
            
            self.results['data_generation_test'] = f'FAIL: {str(e)}'
            logger.error(f"❌ Data generation test failed: {e}")
            raise
    
    def test_memory_usage(self):
        """Test and visualize memory usage during operations."""
        logger.info("Testing memory usage...")
        try:
            memory_usage = []
            timestamps = []
            
            # Get baseline
            mem_start = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
            timestamps.append(0)
            memory_usage.append(mem_start)
            
            # Load model
            agent = Agent(model_path=self.model_path)
            mem_after_load = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            timestamps.append(1)
            memory_usage.append(mem_after_load)
            
            # Run simulations
            env = Chess_Env()
            agent.state = env.board.fen()
            agent.run_simulations(n=20)
            mem_after_sim = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            timestamps.append(2)
            memory_usage.append(mem_after_sim)
            
            # Generate some moves
            for i in range(5):
                move = agent.get_move(env)
                env.push(move)
            mem_after_moves = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            timestamps.append(3)
            memory_usage.append(mem_after_moves)
            
            # Plot memory usage
            plt.figure(figsize=(10, 6))
            plt.plot(timestamps, memory_usage, marker='o', linestyle='-', linewidth=2)
            plt.xticks(timestamps, ['Baseline', 'Model Load', 'MCTS Sim', 'Game Play'])
            plt.ylabel('Memory Usage (MB)')
            plt.xlabel('Operation')
            plt.title('Memory Usage During Chess DRL Operations')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('test_results/memory_usage.png')
            
            # Report on memory growth
            mem_growth = mem_after_moves - mem_start
            mem_model = mem_after_load - mem_start
            mem_mcts = mem_after_sim - mem_after_load
            
            self.results['memory_test'] = {
                'baseline_mb': mem_start,
                'model_load_mb': mem_model,
                'mcts_sim_mb': mem_mcts,
                'total_growth_mb': mem_growth
            }
            
            logger.info(f"✅ Memory test completed: +{mem_growth:.1f}MB ({mem_start:.1f}MB → {mem_after_moves:.1f}MB)")
            logger.info(f"   Model loading: +{mem_model:.1f}MB, MCTS: +{mem_mcts:.1f}MB")
        
        except Exception as e:
            self.results['memory_test'] = f'FAIL: {str(e)}'
            logger.error(f"❌ Memory usage test failed: {e}")
    
    def test_cpu_usage(self):
        """Test and visualize CPU usage during operations."""
        logger.info("Testing CPU usage...")
        try:
            cpu_usage = []
            timestamps = []
            labels = []
            
            # Get baseline
            process = psutil.Process(os.getpid())
            cpu_start = process.cpu_percent(interval=0.1)
            timestamps.append(0)
            cpu_usage.append(cpu_start)
            labels.append('Baseline')
            
            # During model inference
            agent = Agent(model_path=self.model_path)
            env = Chess_Env()
            
            # Measure during simulations
            start_time = time.time()
            process.cpu_percent(interval=None)  # Reset measurement
            agent.run_simulations(n=30)
            sim_cpu = process.cpu_percent(interval=None) / psutil.cpu_count()  # Normalize by core count
            timestamps.append(len(timestamps))
            cpu_usage.append(sim_cpu)
            labels.append('MCTS Sim')
            sim_time = time.time() - start_time
            
            # Measure during batch processing
            batch_size = 32
            fake_states = torch.randn(batch_size, config.amount_of_input_planes, 8, 8).to(self.device)
            start_time = time.time()
            process.cpu_percent(interval=None)  # Reset measurement
            for _ in range(10):  # Multiple forward passes
                with torch.no_grad():
                    _, _ = agent.model(fake_states)
            batch_cpu = process.cpu_percent(interval=None) / psutil.cpu_count()
            timestamps.append(len(timestamps))
            cpu_usage.append(batch_cpu)
            labels.append('Batch Process')
            batch_time = time.time() - start_time
            
            # Plot CPU usage
            plt.figure(figsize=(10, 6))
            plt.bar(timestamps, cpu_usage, width=0.6)
            plt.xticks(timestamps, labels)
            plt.ylabel('CPU Usage (% per core)')
            plt.xlabel('Operation')
            plt.title('CPU Usage During Chess DRL Operations')
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.savefig('test_results/cpu_usage.png')
            
            self.results['cpu_test'] = {
                'baseline_pct': cpu_start,
                'mcts_simulation_pct': sim_cpu,
                'mcts_simulation_time': sim_time,
                'batch_processing_pct': batch_cpu,
                'batch_processing_time': batch_time
            }
            
            logger.info(f"✅ CPU test completed:")
            logger.info(f"   MCTS simulation: {sim_cpu:.1f}% CPU over {sim_time:.2f}s")
            logger.info(f"   Batch processing: {batch_cpu:.1f}% CPU over {batch_time:.2f}s")
        
        except Exception as e:
            self.results['cpu_test'] = f'FAIL: {str(e)}'
            logger.error(f"❌ CPU usage test failed: {e}")
    
    def test_gpu_usage(self):
        """Test and visualize GPU usage if available."""
        if not torch.cuda.is_available():
            logger.info("Skipping GPU test - CUDA not available")
            self.results['gpu_test'] = 'SKIPPED'
            return
        
        logger.info("Testing GPU usage...")
        try:
            import pynvml
            pynvml.nvmlInit()
            
            # Get device handle for first GPU
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Get baseline
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            baseline_used = info.used / 1024 / 1024  # MB
            
            # Test batch processing with increasing batch sizes
            batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
            mem_usages = []
            util_percentages = []
            
            for batch_size in batch_sizes:
                # Clear cache
                torch.cuda.empty_cache()
                
                # Create batch
                fake_states = torch.randn(batch_size, config.amount_of_input_planes, 8, 8).to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    _, _ = self.test_model(fake_states)
                
                # Get memory usage after batch
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_usage = info.used / 1024 / 1024  # MB
                mem_usages.append(mem_usage)
                
                # Get GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                util_percentages.append(utilization.gpu)
            
            # Plot GPU memory usage
            plt.figure(figsize=(12, 10))
            
            # Memory usage subplot
            plt.subplot(2, 1, 1)
            plt.plot(batch_sizes, mem_usages, marker='o', linestyle='-', linewidth=2)
            plt.xlabel('Batch Size')
            plt.ylabel('GPU Memory Usage (MB)')
            plt.title('GPU Memory Usage vs Batch Size')
            plt.grid(True)
            plt.xscale('log', base=2)
            
            # Utilization subplot
            plt.subplot(2, 1, 2)
            plt.plot(batch_sizes, util_percentages, marker='s', linestyle='-', linewidth=2, color='orange')
            plt.xlabel('Batch Size')
            plt.ylabel('GPU Utilization (%)')
            plt.title('GPU Utilization vs Batch Size')
            plt.grid(True)
            plt.xscale('log', base=2)
            
            plt.tight_layout()
            plt.savefig('test_results/gpu_usage.png')
            
            # Clean up
            torch.cuda.empty_cache()
            pynvml.nvmlShutdown()
            
            self.results['gpu_test'] = {
                'baseline_mb': baseline_used,
                'batch_sizes': batch_sizes,
                'memory_usage_mb': mem_usages,
                'utilization_pct': util_percentages,
            }
            
            logger.info(f"✅ GPU test completed:")
            logger.info(f"   Baseline memory usage: {baseline_used:.1f}MB")
            logger.info(f"   Memory range: {min(mem_usages):.1f}MB - {max(mem_usages):.1f}MB")
            logger.info(f"   Max utilization: {max(util_percentages)}%")
        
        except Exception as e:
            self.results['gpu_test'] = f'FAIL: {str(e)}'
            logger.error(f"❌ GPU usage test failed: {e}")
    
    def test_simulation_speed(self):
        """Test and benchmark MCTS simulation speed."""
        logger.info("Testing simulation speed...")
        try:
            agent = Agent(model_path=self.model_path)
            env = Chess_Env()
            agent.state = env.board.fen()
            
            # Test simulation counts
            sim_counts = [1, 5, 10, 20, 50, 100]
            times = []
            
            for count in sim_counts:
                # Time simulations
                start_time = time.time()
                agent.run_simulations(n=count)
                elapsed = time.time() - start_time
                times.append(elapsed)
            
            # Calculate simulations per second
            sims_per_sec = [count/time for count, time in zip(sim_counts, times)]
            
            # Plot simulation speed
            plt.figure(figsize=(10, 6))
            plt.plot(sim_counts, sims_per_sec, marker='o', linestyle='-', linewidth=2)
            plt.xlabel('Number of Simulations')
            plt.ylabel('Simulations per Second')
            plt.title('MCTS Simulation Speed')
            plt.grid(True)
            plt.axhline(y=np.mean(sims_per_sec), color='r', linestyle='--', 
                       label=f'Avg: {np.mean(sims_per_sec):.1f} sim/s')
            plt.legend()
            plt.tight_layout()
            plt.savefig('test_results/simulation_speed.png')
            
            self.results['simulation_speed_test'] = {
                'sim_counts': sim_counts,
                'times': times,
                'sims_per_sec': sims_per_sec,
                'avg_sims_per_sec': np.mean(sims_per_sec)
            }
            
            logger.info(f"✅ Simulation speed test completed:")
            logger.info(f"   Average speed: {np.mean(sims_per_sec):.1f} simulations/second")
            
            # Estimate time for full game
            moves_per_game = 40  # Approximate number of moves per player in a typical game
            sims_per_move = config.SIMULATIONS_PER_MOVE
            total_sims = moves_per_game * 2 * sims_per_move  # Both players
            est_game_time = total_sims / np.mean(sims_per_sec)
            
            logger.info(f"   Estimated time for a full game ({moves_per_game*2} moves, "
                       f"{sims_per_move} sims/move): {est_game_time/60:.1f} minutes")
        
        except Exception as e:
            self.results['simulation_speed_test'] = f'FAIL: {str(e)}'
            logger.error(f"❌ Simulation speed test failed: {e}")
    
    def test_batch_processing(self):
        """Test batch processing speed for training."""
        logger.info("Testing batch processing speed...")
        try:
            # Test different batch sizes
            batch_sizes = [8, 16, 32, 64, 128, 256]
            times = []
            
            for batch_size in batch_sizes:
                # Create random input batch
                fake_states = torch.randn(batch_size, config.amount_of_input_planes, 8, 8).to(self.device)
                fake_policies = torch.randn(batch_size, config.OUTPUT_SHAPE[0]).to(self.device)
                fake_values = torch.randn(batch_size, config.OUTPUT_SHAPE[1]).to(self.device)
                
                # Warm-up
                for _ in range(3):
                    with torch.no_grad():
                        self.test_model(fake_states)
                
                # Time forward + backward
                self.test_model.train()
                start_time = time.time()
                
                # Simulate training steps
                for _ in range(5):  # Multiple iterations for more stable measurement
                    policy_pred, value_pred = self.test_model(fake_states)
                    loss_policy = torch.nn.functional.mse_loss(policy_pred, fake_policies)
                    loss_value = torch.nn.functional.mse_loss(value_pred, fake_values)
                    loss = loss_policy + loss_value
                    loss.backward()
                
                elapsed = (time.time() - start_time) / 5  # Average time per iteration
                times.append(elapsed)
                
                # Clean up for next iteration
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Calculate examples per second
            examples_per_sec = [batch_size/time for batch_size, time in zip(batch_sizes, times)]
            
            # Plot batch processing speed
            plt.figure(figsize=(10, 6))
            plt.plot(batch_sizes, examples_per_sec, marker='o', linestyle='-', linewidth=2)
            plt.xlabel('Batch Size')
            plt.ylabel('Examples per Second')
            plt.title('Training Speed vs Batch Size')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('test_results/batch_processing_speed.png')
            
            # Find optimal batch size (highest examples/sec)
            optimal_idx = np.argmax(examples_per_sec)
            optimal_batch = batch_sizes[optimal_idx]
            
            self.results['batch_processing_test'] = {
                'batch_sizes': batch_sizes,
                'times': times,
                'examples_per_sec': examples_per_sec,
                'optimal_batch_size': optimal_batch
            }
            
            logger.info(f"✅ Batch processing test completed:")
            logger.info(f"   Optimal batch size: {optimal_batch}")
            logger.info(f"   Processing speed at optimal batch: {examples_per_sec[optimal_idx]:.1f} examples/second")
        
        except Exception as e:
            self.results['batch_processing_test'] = f'FAIL: {str(e)}'
            logger.error(f"❌ Batch processing test failed: {e}")
    
    def generate_test_report(self):
        """Generate an HTML report with test results and recommendations."""
        try:
            # Calculate overall status
            status_counts = {}
            for key, value in self.results.items():
                if not key.endswith('_test'):
                    continue
                    
                if isinstance(value, str):
                    status = value.split(':')[0] if ':' in value else value
                else:
                    status = 'PASS'
                
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Prepare report data
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            device_info = f"{self.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})"
            
            # Save results as JSON
            with open('test_results/latest_results.json', 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'device': str(self.device),
                    'results': self.results
                }, f, indent=2, default=str)
            
            # Determine if ready for training
            critical_tests = ['environment_test', 'agent_test', 'mcts_test', 'model_test']
            critical_passed = all(
                self.results.get(test, '').startswith('PASS') 
                for test in critical_tests if test in self.results
            )
            
            # Recommendations
            recommendations = []
            
            if not critical_passed:
                recommendations.append("❌ Fix critical test failures before starting training")
            else:
                recommendations.append("✅ Core components are working correctly")
            
            if 'simulation_speed_test' in self.results and isinstance(self.results['simulation_speed_test'], dict):
                sims_per_sec = self.results['simulation_speed_test'].get('avg_sims_per_sec', 0)
                if sims_per_sec < 10:
                    recommendations.append(f"⚠️ Simulation speed is slow ({sims_per_sec:.1f} sims/sec). Consider reducing SIMULATIONS_PER_MOVE")
                elif sims_per_sec > 50:
                    recommendations.append(f"✅ Good simulation speed ({sims_per_sec:.1f} sims/sec)")
            
            if 'batch_processing_test' in self.results and isinstance(self.results['batch_processing_test'], dict):
                optimal_batch = self.results['batch_processing_test'].get('optimal_batch_size', 0)
                if optimal_batch > 0:
                    recommendations.append(f"💡 Consider using batch size of {optimal_batch} for optimal training speed")
            
            if 'memory_test' in self.results and isinstance(self.results['memory_test'], dict):
                mem_growth = self.results['memory_test'].get('total_growth_mb', 0)
                if mem_growth > 1000:
                    recommendations.append(f"⚠️ High memory usage growth ({mem_growth:.1f} MB). Monitor memory during training")
            
            if torch.cuda.is_available() and 'gpu_test' in self.results and isinstance(self.results['gpu_test'], dict):
                util = self.results['gpu_test'].get('utilization_pct', [0])
                if isinstance(util, list) and len(util) > 0 and max(util) < 50:
                    recommendations.append("⚠️ Low GPU utilization. Consider increasing batch size or model complexity")
            
            # Generate HTML report
            html_report = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Chess DRL Test Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #333; }}
                    .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    .pass {{ color: green; }}
                    .fail {{ color: red; }}
                    .warn {{ color: orange; }}
                    .skipped {{ color: gray; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .recommendations {{ background-color: #e6f7ff; padding: 15px; border-radius: 5px; }}
                    .images {{ display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px; }}
                    .image-container {{ max-width: 45%; }}
                    img {{ max-width: 100%; border: 1px solid #ddd; }}
                </style>
            </head>
            <body>
                <h1>Chess DRL Test Report</h1>
                <div class="summary">
                    <p><strong>Time:</strong> {timestamp}</p>
                    <p><strong>Device:</strong> {device_info}</p>
                    <p><strong>Overall Status:</strong> 
                        {"<span class='pass'>READY FOR TRAINING</span>" if critical_passed else "<span class='fail'>NOT READY - FIX ERRORS</span>"}
                    </p>
                    <p><strong>Test Summary:</strong> 
                        {status_counts.get('PASS', 0)} passed, 
                        {status_counts.get('FAIL', 0)} failed, 
                        {status_counts.get('SKIPPED', 0)} skipped
                    </p>
                </div>
                
                <h2>Test Results</h2>
                <table>
                    <tr>
                        <th>Test</th>
                        <th>Result</th>
                        <th>Details</th>
                    </tr>
            """
            
            for key, value in self.results.items():
                if not key.endswith('_test'):
                    continue
                    
                test_name = key.replace('_test', '').replace('_', ' ').title()
                if isinstance(value, str):
                    if value.startswith('PASS'):
                        result = f"<span class='pass'>PASS</span>"
                        details = value[5:] if len(value) > 5 else ""
                    elif value.startswith('FAIL'):
                        result = f"<span class='fail'>FAIL</span>"
                        details = value[5:] if len(value) > 5 else ""
                    elif value.startswith('SKIPPED'):
                        result = f"<span class='skipped'>SKIPPED</span>"
                        details = value[8:] if len(value) > 8 else ""
                    else:
                        result = value
                        details = ""
                else:
                    result = f"<span class='pass'>PASS</span>"
                    if isinstance(value, dict):
                        details = "<ul>"
                        for k, v in value.items():
                            if not isinstance(v, (list, dict)):
                                details += f"<li>{k}: {v}</li>"
                        details += "</ul>"
                    else:
                        details = str(value)
                
                html_report += f"""
                    <tr>
                        <td>{test_name}</td>
                        <td>{result}</td>
                        <td>{details}</td>
                    </tr>
                """
            
            html_report += f"""
                </table>
                
                <h2>Recommendations</h2>
                <div class="recommendations">
                    <ul>
                        {"".join(f"<li>{rec}</li>" for rec in recommendations)}
                    </ul>
                </div>
                
                <h2>Performance Graphs</h2>
                <div class="images">
                    <div class="image-container">
                        <h3>Memory Usage</h3>
                        <img src="memory_usage.png" alt="Memory Usage Graph">
                    </div>
                    <div class="image-container">
                        <h3>CPU Usage</h3>
                        <img src="cpu_usage.png" alt="CPU Usage Graph">
                    </div>
                    <div class="image-container">
                        <h3>Simulation Speed</h3>
                        <img src="simulation_speed.png" alt="Simulation Speed Graph">
                    </div>
                    <div class="image-container">
                        <h3>Batch Processing</h3>
                        <img src="batch_processing_speed.png" alt="Batch Processing Graph">
                    </div>
                    {"<div class='image-container'><h3>GPU Usage</h3><img src='gpu_usage.png' alt='GPU Usage Graph'></div>" 
                     if torch.cuda.is_available() else ""}
                </div>
                
                <hr>
                <p>Generated by Chess-DRL Test Suite</p>
            </body>
            </html>
            """
            
            # Save HTML report
            with open('test_results/test_report.html', 'w') as f:
                f.write(html_report)
                
            logger.info(f"Test report generated: test_results/test_report.html")
        
        except Exception as e:
            logger.error(f"Failed to generate test report: {e}")

def create_training_monitor():
    """
    Create a class to monitor and visualize training metrics.
    """
    class TrainingMonitor:
        def __init__(self, save_dir='./plots'):
            self.save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)
            
            # Initialize metrics
            self.epochs = []
            self.policy_losses = []
            self.value_losses = []
            self.total_losses = []
            self.mem_usage = []
            self.cpu_usage = []
            self.gpu_usage = []
            self.batch_times = []
            self.replay_buffer_size = []
            self.replay_buffer_values = []  # Average value in buffer
            
            # Resource usage
            self.process = psutil.Process(os.getpid())
            
            # Initialize GPU monitoring if available
            self.gpu_handle = None
            try:
                if torch.cuda.is_available():
                    import pynvml
                    pynvml.nvmlInit()
                    self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                pass
                
        def update(self, epoch, policy_loss, value_loss, total_loss, batch_time, 
                  replay_buffer_size=None, replay_buffer_values=None):
            """Record metrics for one training epoch."""
            self.epochs.append(epoch)
            self.policy_losses.append(policy_loss)
            self.value_losses.append(value_loss)
            self.total_losses.append(total_loss)
            self.batch_times.append(batch_time)
            
            # Add resource usage
            self.mem_usage.append(self.process.memory_info().rss / 1024 / 1024)
            self.cpu_usage.append(self.process.cpu_percent() / psutil.cpu_count())
            
            # GPU usage if available
            gpu_util = 0
            if self.gpu_handle:
                try:
                    import pynvml
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    gpu_util = utilization.gpu
                except:
                    pass
            self.gpu_usage.append(gpu_util)
            
            # Replay buffer stats
            if replay_buffer_size is not None:
                self.replay_buffer_size.append(replay_buffer_size)
            if replay_buffer_values is not None:
                self.replay_buffer_values.append(replay_buffer_values)
                
            # Update plots every 10 epochs or on the first epoch
            if epoch == 1 or epoch % 10 == 0:
                self.plot()
                
        def plot(self):
            """Create and save plots of the training metrics."""
            # Create figure with 2x3 subplots
            fig, axs = plt.subplots(2, 3, figsize=(18, 12))
            
            # Plot loss curves
            axs[0, 0].plot(self.epochs, self.policy_losses, label='Policy Loss')
            axs[0, 0].plot(self.epochs, self.value_losses, label='Value Loss')
            axs[0, 0].plot(self.epochs, self.total_losses, label='Total Loss')
            axs[0, 0].set_title('Training Loss')
            axs[0, 0].set_xlabel('Epoch')
            axs[0, 0].set_ylabel('Loss')
            axs[0, 0].legend()
            axs[0, 0].grid(True)
            
            # Plot resource usage
            ax1 = axs[0, 1]
            ax1.plot(self.epochs, self.mem_usage, 'b-', label='Memory (MB)')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Memory Usage (MB)', color='b')
            ax1.tick_params('y', colors='b')
            ax1.grid(True)
            
            ax2 = ax1.twinx()
            ax2.plot(self.epochs, self.cpu_usage, 'r-', label='CPU (%)')
            ax2.plot(self.epochs, self.gpu_usage, 'g-', label='GPU (%)')
            ax2.set_ylabel('Usage (%)', color='r')
            ax2.tick_params('y', colors='r')
            ax2.legend(loc='upper right')
            axs[0, 1].set_title('Resource Usage')
            
            # Plot batch times
            axs[0, 2].plot(self.epochs, self.batch_times, 'g-')
            axs[0, 2].set_title('Batch Processing Time')
            axs[0, 2].set_xlabel('Epoch')
            axs[0, 2].set_ylabel('Time (seconds)')
            axs[0, 2].grid(True)
            
            # Plot replay buffer size
            if self.replay_buffer_size:
                axs[1, 0].plot(self.epochs[:len(self.replay_buffer_size)], self.replay_buffer_size, 'b-')
                axs[1, 0].set_title('Replay Buffer Size')
                axs[1, 0].set_xlabel('Epoch')
                axs[1, 0].set_ylabel('Number of Examples')
                axs[1, 0].get_yaxis().set_major_formatter(
                    plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
                axs[1, 0].grid(True)
            
            # Plot replay buffer value distribution
            if self.replay_buffer_values and len(self.replay_buffer_values) > 0:
                latest_values = self.replay_buffer_values[-1]
                if isinstance(latest_values, (list, np.ndarray)) and len(latest_values) > 0:
                    axs[1, 1].hist(latest_values, bins=21, range=(-1, 1), alpha=0.7)
                    axs[1, 1].set_title('Value Distribution in Replay Buffer')
                    axs[1, 1].set_xlabel('Value')
                    axs[1, 1].set_ylabel('Count')
                    axs[1, 1].grid(True)
            
            # Plot learning rate if available
            if hasattr(self, 'learning_rates') and len(self.learning_rates) > 0:
                axs[1, 2].plot(self.epochs[:len(self.learning_rates)], self.learning_rates, 'r-')
                axs[1, 2].set_title('Learning Rate')
                axs[1, 2].set_xlabel('Epoch')
                axs[1, 2].set_ylabel('Learning Rate')
                axs[1, 2].grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'training_metrics.png'))
            plt.close()
            
        def close(self):
            """Clean up resources."""
            try:
                if torch.cuda.is_available():
                    import pynvml
                    pynvml.nvmlShutdown()
            except:
                pass
                
    return TrainingMonitor

def add_monitoring_to_train(replay_buffer):
    """Generate code that adds monitoring to the training process."""
    monitor_code = """
# Add to top of train.py
from test import create_training_monitor

# Add in main() after initializing optimizer
training_monitor = create_training_monitor()(save_dir=config.LOSS_PLOTS_FOLDER)

# Add in training loop after calculating loss
batch_time = time.time() - batch_start_time
training_monitor.update(
    epoch=epoch+1,
    policy_loss=loss_policy.item(),
    value_loss=loss_value.item(), 
    total_loss=loss.item(),
    batch_time=batch_time,
    replay_buffer_size=len(replay_buffer),
    replay_buffer_values=np.array([v for _, _, v in random.sample(replay_buffer, min(1000, len(replay_buffer)))])
)

# Add at end of main()
training_monitor.close()
"""
    return monitor_code

def main():
    parser = argparse.ArgumentParser(description="Run tests for chess DRL project")
    parser.add_argument("--model", type=str, help="Path to model for testing")
    args = parser.parse_args()
    
    # Create test_results directory if it doesn't exist
    os.makedirs('test_results', exist_ok=True)
    
    # Run test suite
    test_suite = TestSuite(model_path=args.model)
    results = test_suite.run_all_tests()
    
    # Output summary
    critical_tests = ['environment_test', 'agent_test', 'mcts_test', 'model_test']
    critical_passed = all(
        results.get(test, '').startswith('PASS') 
        for test in critical_tests if test in results
    )
    
    if critical_passed:
        logger.info("\n✅ READY FOR TRAINING: All critical tests passed!")
        logger.info("Open test_results/test_report.html to see detailed performance analysis")
    else:
        logger.info("\n⚠️ NOT READY FOR TRAINING: Critical tests failed")
        logger.info("Please fix the issues reported in test_results/test_report.html")
    
    # Provide code for adding monitoring to train.py
    from train import replay_buffer
    monitor_code = add_monitoring_to_train(replay_buffer)
    with open('test_results/monitoring_code.py', 'w') as f:
        f.write(monitor_code)
    
    logger.info("\nTo add resource monitoring and visualization to your training:")
    logger.info("Check test_results/monitoring_code.py for code to add to train.py")

if __name__ == "__main__":
    main()