"""Save selected checkpoints and optionally stop before the scheduler horizon."""
from transformers import TrainerCallback


class CheckpointScheduleCallback(TrainerCallback):
    def __init__(self, checkpoint_steps=None, stop_after_steps=None):
        self.steps=set(checkpoint_steps) if checkpoint_steps is not None else None
        self.stop_after_steps=stop_after_steps

    def on_step_end(self,args,state,control,**kwargs):
        if self.steps is not None:
            control.should_save=state.global_step in self.steps
        if self.stop_after_steps is not None and state.global_step>=self.stop_after_steps:
            control.should_training_stop=True
            control.should_save=True
        return control


def effective_steps(config):
    maximum=config["max_steps"]
    stop=config.get("stop_after_steps",maximum)
    if type(stop) is not int or stop<1 or stop>maximum:
        raise ValueError("stop_after_steps must be an integer between 1 and max_steps")
    selected=config.get("checkpoint_steps")
    if selected is not None and (not selected or any(type(step) is not int or step<1 or step>stop for step in selected)):
        raise ValueError("checkpoint_steps must be positive integers no greater than the actual stop step")
    if selected is not None and not config.get("save_final_model",True) and stop not in selected:
        raise ValueError("Include the final step in checkpoint_steps when save_final_model is false")
    return stop
