import pytest
import torch
from camix_director.config import ModelConfig
from camix_director.model import MemoryDirector
from camix_director.export import example_inputs

@pytest.fixture
def setup():
    torch.manual_seed(2);torch.set_num_threads(2)
    cfg=ModelConfig(max_cameras=3,image_size=16,audio_samples=256,hidden_dim=16,heads=4,layers=1,memory_tokens=4,dropout=0)
    model=MemoryDirector(cfg).eval()
    return model,list(example_inputs(cfg))

def test_value_contract_and_gain(setup):
    model,args=setup;keep,take,gain,memory,mask=model(*args)
    assert keep.shape==(1,1) and take.shape==(1,3) and memory.shape==(1,4,16)
    torch.testing.assert_close(gain,take-keep)
    assert gain[0,0]==0 and mask[0,-1]==1

def test_camera_permutation_equivariance(setup):
    model,args=setup;expected=model(*args);p=torch.tensor([2,0,1]);permuted=list(args)
    for i in range(6):permuted[i]=args[i].index_select(1,p)
    actual=model(*permuted)
    torch.testing.assert_close(actual[0],expected[0],atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(actual[1],expected[1].index_select(1,p),atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(actual[-2],expected[-2],atol=2e-6,rtol=2e-5)

def test_all_cameras_unavailable_no_nan(setup):
    model,args=setup;args[2].zero_();args[3].zero_()
    result=model(*args)
    assert all(torch.isfinite(x).all() for x in result)
    assert (result[1]<-999).all()

def test_bounded_memory_input_not_mutated(setup):
    model,args=setup;old=args[-2].clone()
    for _ in range(9):
        output=model(*args);args[-2:]=output[-2:]
        assert output[-2].shape==(1,4,16)
    assert (output[-1]==1).all()
    assert not torch.equal(old,args[-2])

def test_memory_changes_model_outputs_not_quality_claim(setup):
    model,args=setup;full=model(*args)
    args[-2:]=model.initial_memory();empty=model(*args)
    assert not torch.allclose(full[1],empty[1],atol=1e-6)

def test_raw_video_and_audio_receive_gradients(setup):
    model,args=setup;args[0]=args[0].clone().requires_grad_();args[1]=args[1].clone().requires_grad_()
    output=model(*args);(output[0].sum()+output[1][:,1:].sum()).backward()
    assert args[0].grad.abs().sum()>0 and args[1].grad.abs().sum()>0

def test_repeated_identical_step_deterministic(setup):
    model,args=setup
    a=model(*args);b=model(*args)
    for x,y in zip(a,b):torch.testing.assert_close(x,y)

def test_future_cannot_change_past_prefix(setup):
    model,args=setup
    def replay(future):
        memory,mask=model.initial_memory();out=[]
        for i in range(6):
            current=list(args);current[-2:]=[memory,mask]
            if i>=3:current[0]=current[0]*future
            result=model(*current);memory,mask=result[-2:];out.append(result[1].detach())
        return out
    a,b=replay(1),replay(-1)
    for x,y in zip(a[:3],b[:3]):torch.testing.assert_close(x,y)

def test_invalid_config():
    with pytest.raises(ValueError):ModelConfig(hidden_dim=17,heads=4)

def test_normalized_audio_is_finite_for_silence_and_quiet_audio():
    from camix_director.model import NormalizedAudioEncoder
    encoder=NormalizedAudioEncoder(16)
    silence=torch.zeros(2,1,4000,requires_grad=True)
    result=encoder(silence)
    assert result.shape==(2,16) and torch.isfinite(result).all()
    result.sum().backward();assert torch.isfinite(silence.grad).all()

def test_normalized_audio_keeps_learnable_energy_information():
    from camix_director.model import NormalizedAudioEncoder
    torch.manual_seed(3);encoder=NormalizedAudioEncoder(16)
    wave=torch.sin(torch.arange(4000)*.04).reshape(1,1,-1)
    quiet=encoder(wave*.0001);loud=encoder(wave*.01)
    assert torch.isfinite(quiet).all() and not torch.allclose(quiet,loud,atol=1e-4)
