import torch


def wasserstein_generator_loss(generated_examples, critic):
    """
    The generator is optimized to minimize the critic prediction
    """
    critic_eval = torch.mean(critic(*generated_examples))
    return -1 * critic_eval


def wasserstein_critic_loss(generated_examples, real_examples, critic):
    """
    The discrimator is optimized to maximize the difference in prediction between the real examples and the generated examples
    """
    critic_eval_generated_examples = critic(*generated_examples)
    critic_eval_real_examples = critic(*real_examples)
    return -1 * (torch.mean(critic_eval_real_examples) - torch.mean(critic_eval_generated_examples))
