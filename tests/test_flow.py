import torch

from neosim.models.flow import NeoFlow, NeoFlowConfig, flow_matching_loss, sample


def test_forward_and_loss():
    torch.manual_seed(0)
    cfg = NeoFlowConfig(d_model=48, n_layers=2, n_heads=4, max_seq_len=32, dropout=0.0)
    model = NeoFlow(cfg)
    b, t = 3, 16
    x = torch.randn(b, t, 3)
    static = torch.randn(b, 13)
    events = torch.zeros(b, t, 16)
    time = torch.rand(b, t, 1)
    mask = torch.ones(b, t, 3)
    loss, parts = flow_matching_loss(model, x, mask, static, events, time, cond_dropout=0.1)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert "loss_ch0" in parts
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_sampling_with_prefix_context():
    torch.manual_seed(1)
    cfg = NeoFlowConfig(d_model=32, n_layers=1, n_heads=4, max_seq_len=32)
    model = NeoFlow(cfg).eval()
    b, t = 1, 16
    static = torch.randn(b, 13)
    events = torch.zeros(b, t, 16)
    time = torch.rand(b, t, 1)
    mask = torch.ones(b, t, 3)
    context = torch.zeros_like(time).expand(-1, -1, 3)
    context_mask = torch.zeros_like(context)
    context_mask[:, :4] = 1
    out = sample(model, static, events, time, mask, context, context_mask, steps=3, seed=0)
    assert out.shape == (b, t, 3)
    assert torch.allclose(out[:, :4], context[:, :4])


def test_auxiliary_structure_losses_are_finite():
    torch.manual_seed(2)
    cfg = NeoFlowConfig(d_model=48, n_layers=2, n_heads=4, max_seq_len=32, dropout=0.0)
    model = NeoFlow(cfg)
    b, t = 4, 20
    x = torch.randn(b, t, 3)
    static = torch.randn(b, 13)
    events = torch.zeros(b, t, 16)
    time = torch.rand(b, t, 1)
    mask = torch.ones(b, t, 3)
    loss, parts = flow_matching_loss(
        model,
        x,
        mask,
        static,
        events,
        time,
        aux_moment=0.05,
        aux_acf=0.5,
        aux_corr=0.5,
    )
    assert torch.isfinite(loss)
    assert "aux_moment" in parts
    loss.backward()


def test_channel_specific_sampling_noise():
    torch.manual_seed(3)
    cfg = NeoFlowConfig(d_model=32, n_layers=1, n_heads=4, max_seq_len=32)
    model = NeoFlow(cfg).eval()
    b, t = 2, 16
    static = torch.randn(b, 13)
    events = torch.zeros(b, t, 16)
    time = torch.rand(b, t, 1)
    mask = torch.ones(b, t, 3)
    out = sample(
        model,
        static,
        events,
        time,
        mask,
        steps=2,
        seed=0,
        noise_phi_channels=(0.1, 0.2, 0.3),
        noise_scale_channels=(0.5, 0.6, 0.7),
    )
    assert out.shape == (b, t, 3)
    assert torch.isfinite(out).all()
