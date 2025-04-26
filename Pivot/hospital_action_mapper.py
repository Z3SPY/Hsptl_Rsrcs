def map_action(action_idx, scenario, model):
    # 0: no-op
    if action_idx == 0:
        return
    # 1: +1 triage
    elif action_idx == 1:
        if scenario.n_triage < MAX:
            scenario.n_triage += 1
            model.args.triage.put(CustomResource(model.env, 1))
    # etc. for each action
