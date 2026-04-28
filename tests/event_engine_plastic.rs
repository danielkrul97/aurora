use aurora::engine::EventNetwork;
use aurora::lif::LifParams;
use aurora::stdp::StdpParams;

#[test]
fn non_plastic_synapse_keeps_weight_under_stdp() {
    let mut net = EventNetwork::new();
    net.stdp_params = StdpParams {
        a_plus: 0.01,
        a_minus: 0.01,
        tau_plus: 0.020,
        tau_minus: 0.020,
    };

    let pre = net.add_neuron(LifParams::default());
    let post = net.add_neuron(LifParams::default());

    let initial = 0.030;
    let plastic = net.add_synapse(pre, post, None, initial, 0.001);
    let fixed = net.add_synapse(pre, post, None, initial, 0.001);
    net.synapses[fixed.0].plastic = false;

    let amp = 0.030;
    let n_pairs = 20;
    for i in 0..n_pairs {
        let t_pre = (i as f64) * 0.050 + 0.001;
        let t_post = t_pre + 0.005;
        net.schedule_input(t_pre, pre, None, amp);
        net.schedule_input(t_post, post, None, amp);
    }
    net.run_until((n_pairs as f64) * 0.050 + 0.100);

    let final_plastic = net.synapses[plastic.0].weight;
    let final_fixed = net.synapses[fixed.0].weight;

    assert!(
        (final_plastic - initial).abs() > 1e-4,
        "plastic synapse must drift under repeated STDP pairing: {initial} → {final_plastic}"
    );
    assert!(
        (final_fixed - initial).abs() < 1e-12,
        "non-plastic synapse must not drift: {initial} → {final_fixed}"
    );
}
