"""Exactness proof: load the published othello_v0 haiku weights into our Flax
AZNet port and verify identical outputs on real board positions."""
import pickle, numpy as np
import jax, jax.numpy as jnp
from pgx4.az_baseline import load_model  # applies jax.core shims for haiku
from pgx4.train import ActorCritic
from pgx4.elo import random_start
import pgx

theirs = load_model("othello_v0", "baselines")  # downloads the ckpt if missing
d = pickle.load(open("baselines/othello_v0.ckpt", "rb"))
hk_params, hk_state = d["params"], d["state"]

# ---- build flax variables from haiku dicts --------------------------------
P, B = {}, {}
def conv(fname, hname):
    P[fname] = {"kernel": jnp.asarray(hk_params[hname]["w"]),
                "bias": jnp.asarray(hk_params[hname]["b"])}
def dense(fname, hname):
    P[fname] = {"kernel": jnp.asarray(hk_params[hname]["w"]),
                "bias": jnp.asarray(hk_params[hname]["b"])}
def bn(fname, hname):
    P[fname] = {"scale": jnp.asarray(hk_params[hname]["scale"]).reshape(-1),
                "bias": jnp.asarray(hk_params[hname]["offset"]).reshape(-1)}
    B[fname] = {"mean": jnp.asarray(hk_state[hname + "/~/mean_ema"]["average"]).reshape(-1),
                "var": jnp.asarray(hk_state[hname + "/~/var_ema"]["average"]).reshape(-1)}

conv("Conv_0", "az_net/conv2_d")
for i in range(6):
    bn(f"BatchNorm_{2*i}",   f"az_net/block_{i}/batch_norm")
    conv(f"Conv_{2*i+1}",    f"az_net/block_{i}/conv2_d")
    bn(f"BatchNorm_{2*i+1}", f"az_net/block_{i}/batch_norm_1")
    conv(f"Conv_{2*i+2}",    f"az_net/block_{i}/conv2_d_1")
bn("BatchNorm_12", "az_net/batch_norm")
conv("Conv_13", "az_net/conv2_d_1"); bn("BatchNorm_13", "az_net/batch_norm_1")
dense("Dense_0", "az_net/linear")
conv("Conv_14", "az_net/conv2_d_2"); bn("BatchNorm_14", "az_net/batch_norm_2")
dense("Dense_1", "az_net/linear_1"); dense("Dense_2", "az_net/linear_2")
variables = {"params": P, "batch_stats": B}

# ---- compare on 256 real board positions ----------------------------------
env = pgx.make("othello")
net = ActorCritic(n_actions=env.num_actions, width=128, depth=6, arch="aznet")
ref = net.init(jax.random.PRNGKey(0), jnp.zeros((1,) + env.observation_shape))
# structural check: same tree structure as a fresh init
import jax.tree_util as tu
assert tu.tree_structure(ref) == tu.tree_structure(variables), "tree mismatch"

states = jax.vmap(lambda k: random_start(env, k, 30))(jax.random.split(jax.random.PRNGKey(7), 256))
obs = states.observation.astype(jnp.float32)

ours_logits, ours_value = net.apply(variables, obs)          # train=False default
th_logits, th_value = theirs(obs)

dl = float(jnp.abs(ours_logits - th_logits).max())
dv = float(jnp.abs(ours_value - th_value.reshape(-1)).max())
same_move = float((jnp.argmax(jnp.where(states.legal_action_mask, ours_logits, -1e9), -1)
                   == jnp.argmax(jnp.where(states.legal_action_mask, th_logits, -1e9), -1)).mean())
print(f"max |logit diff| = {dl:.2e}")
print(f"max |value diff| = {dv:.2e}")
print(f"greedy-move agreement over 256 positions: {same_move:.4f}")
