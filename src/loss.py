import torch
import torch.nn as nn


def compute_td_loss(states, actions, rewards, next_states, is_done,
                    agent, target_agent,
                    gamma=0.99,
                    check_shapes=False,
                    device='cpu'):
    """ Compute td loss using torch operations only. Use the formulae above. """
    states = torch.tensor(states, device=device, dtype=torch.float)    # shape: [batch_size, *state_shape]

    # for some torch reason should not make actions a tensor
    actions = torch.tensor(actions, device=device, dtype=torch.long)    # shape: [batch_size]
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)   # shape: [batch_size]
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)  # shape: [batch_size, *state_shape]
    is_done = torch.tensor(
        is_done.astype('float32'),
        device=device,
        dtype=torch.float
    )   # shape: [batch_size]
    is_not_done = 1 - is_done

    # get q-values for all actions in current states
    predicted_qvalues = agent(states)
    assert predicted_qvalues.requires_grad, "qvalues must be a torch tensor with grad"

    # compute q-values in next states
    with torch.no_grad():
        target_predicted_next_qvalues = target_agent(next_states).max(dim=1)[0]

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues.gather(1, actions.reshape(-1, 1)).squeeze(1)

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    # you can multiply next state values by is_not_done to achieve this.
    target_qvalues_for_actions = (rewards + gamma*is_not_done*target_predicted_next_qvalues).detach()

    assert target_qvalues_for_actions.requires_grad == False, "do not send gradients to target!"

    # mean squared error loss to minimize
    loss = nn.functional.mse_loss(predicted_qvalues_for_actions, target_qvalues_for_actions)

    if check_shapes:
        assert target_qvalues_for_actions.data.dim() == 1, "there's something wrong with target q-values, they must be a vector"

    return loss
