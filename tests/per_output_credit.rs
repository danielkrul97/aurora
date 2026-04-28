use aurora::eligibility::EligibilityParams;
use aurora::lif::LifParams;
use aurora::network::Network;
use aurora::neuromodulator::ModulatorParams;
use aurora::types::NeuronId;

const DT: f64 = 0.0001;

fn build() -> Network {
    let mut net = Network::new();
    net.eligibility_params = Some(EligibilityParams {
        tau: 0.300,
        learning_rate: 1.0,
    });
    let pre = net.add_neuron(LifParams::default());
    let post0 = net.add_neuron(LifParams::default());
    let post1 = net.add_neuron(LifParams::default());
    net.add_synapse(pre, post0, None, 0.001, 0.001);
    net.add_synapse(pre, post1, None, 0.001, 0.001);
    let mod_params = ModulatorParams {
        decay_tau: 0.150,
        baseline: 0.0,
    };
    net.enable_neuron_modulator(post0, mod_params);
    net.enable_neuron_modulator(post1, mod_params);
    net
}

fn force_pair(net: &mut Network, pre_t: f64, post0_t: f64, post1_t: f64) {
    let amp = 0.030;
    net.schedule_input(pre_t, NeuronId(0), None, amp);
    net.schedule_input(post0_t, NeuronId(1), None, amp);
    net.schedule_input(post1_t, NeuronId(2), None, amp);
}

fn run_for(net: &mut Network, duration: f64) {
    let n = (duration / DT).ceil() as usize;
    for _ in 0..n {
        net.step(DT);
    }
}

#[test]
fn reward_to_one_neuron_grows_only_its_synapse() {
    let mut net = build();
    let initial_w0 = net.synapses[0].weight;
    let initial_w1 = net.synapses[1].weight;
    assert!((initial_w0 - initial_w1).abs() < 1e-12);

    let pair_period = 0.150;
    let n_pairs = 20;
    for i in 0..n_pairs {
        let t_pre = (i as f64) * pair_period + 0.001;
        let t_post = t_pre + 0.004;
        force_pair(&mut net, t_pre, t_post, t_post);
    }
    for i in 0..n_pairs {
        let reward_t = (i as f64) * pair_period + 0.020;
        let dt_to_reward = reward_t - net.now;
        run_for(&mut net, dt_to_reward);
        net.reward_neuron(NeuronId(1), 1.0);
    }
    let final_t = (n_pairs as f64) * pair_period + 0.500;
    let dt_remaining = final_t - net.now;
    run_for(&mut net, dt_remaining);

    let final_w0 = net.synapses[0].weight;
    let final_w1 = net.synapses[1].weight;
    assert!(
        final_w0 > initial_w0 + 0.001,
        "syn[0] (rewarded target) must grow: {initial_w0} → {final_w0}"
    );
    assert!(
        (final_w1 - initial_w1).abs() < 1e-9,
        "syn[1] (unrewarded target) must not drift: {initial_w1} → {final_w1}"
    );
}

#[test]
fn global_modulator_remains_fallback_when_no_per_neuron() {
    let mut net = Network::new();
    net.eligibility_params = Some(EligibilityParams {
        tau: 0.300,
        learning_rate: 1.0,
    });
    net.modulator = Some(aurora::neuromodulator::Modulator::new(ModulatorParams {
        decay_tau: 0.150,
        baseline: 0.0,
    }));
    let pre = net.add_neuron(LifParams::default());
    let post = net.add_neuron(LifParams::default());
    net.add_synapse(pre, post, None, 0.001, 0.001);
    let initial_w = net.synapses[0].weight;

    let pair_period = 0.150;
    let n_pairs = 15;
    for i in 0..n_pairs {
        let t_pre = (i as f64) * pair_period + 0.001;
        let t_post = t_pre + 0.004;
        net.schedule_input(t_pre, NeuronId(0), None, 0.030);
        net.schedule_input(t_post, NeuronId(1), None, 0.030);
    }
    for i in 0..n_pairs {
        let reward_t = (i as f64) * pair_period + 0.020;
        let dt_to_reward = reward_t - net.now;
        run_for(&mut net, dt_to_reward);
        net.reward(1.0);
    }
    let final_t = (n_pairs as f64) * pair_period + 0.500;
    let dt_remaining = final_t - net.now;
    run_for(&mut net, dt_remaining);

    let final_w = net.synapses[0].weight;
    assert!(
        final_w > initial_w + 0.001,
        "global modulator must still drive growth: {initial_w} → {final_w}"
    );
}
