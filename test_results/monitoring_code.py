
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
