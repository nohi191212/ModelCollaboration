import torch
def output_count(method):
    assert method=='three_state'
    return 3
def scores(raw,method):
    assert method=='three_state'
    p=torch.as_tensor(raw).softmax(1)
    return (p[:,0]-p[:,1]).numpy()
