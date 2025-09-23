# compact_embed.py
from trainingUtils import OutOfDistributionTraining
from reservoirTrainingUtils import TrainingToInference
from agent import LinearAgent
from environment import Environment
from reservoir import initialize_reservoir

import numpy as np, matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE

def get_data(agent_mode='normal', rounds=1, episodes=600, time_steps=30):
    agent, env = LinearAgent(), Environment()
    ref_rewards, _, ref_weights = OutOfDistributionTraining(
        agent, env, rounds=rounds, episodes=episodes, time_steps=time_steps,
        mode='normal', verbose=False, return_weights=True
    )
    if agent_mode == 'normal':
        rewards, weights = ref_rewards, ref_weights
    else:
        reservoir = initialize_reservoir()
        rewards, _, _, _, weights = TrainingToInference(
            agent, env, reservoir, rounds=rounds, episodes=episodes,
            time_steps=time_steps, verbose=False
        )
    return np.array(rewards), np.array(weights), np.array(ref_weights)

def flatten4(W):
    P,E,A,D = W.shape; F=A*D
    return W.reshape(P,E,F),(P,E,F)

def alphas_from_rewards(rew, lo=0.05, hi=1.0, bin_size=15):
    P,E=rew.shape; nb=(E+bin_size-1)//bin_size
    pad=np.zeros((P,nb*bin_size)); pad[:,-E:]=rew
    b=pad.reshape(P,nb,bin_size).mean(2)
    mn, mx = b.min(1,keepdims=True), b.max(1,keepdims=True)
    norm=(b-mn)/np.maximum(mx-mn,1e-8)
    a=lo+(hi-lo)*norm
    return np.repeat(a,bin_size,1)[:,-E:]

def make_reducer(name, n_components):
    name=name.lower()
    if name=='pca':    return PCA(n_components=n_components, svd_solver='full')
    if name=='isomap': return Isomap(n_components=n_components, n_neighbors=20)
    if name=='tsne':   return TSNE(n_components=min(n_components,3), init='pca', perplexity=35, learning_rate='auto')
    raise ValueError("reducer ∈ {'pca','isomap','tsne'}")

def fit_transform(weights, ref_weights, reducer='pca', n_components=3, fit_space='reference'):
    Rname=reducer.lower()
    W,(P,E,F)=flatten4(weights)

    if Rname=='tsne':
        if fit_space=='joint':
            Wref,_=flatten4(ref_weights)
            X=np.vstack([Wref.reshape(-1,F), W.reshape(-1,F)])
            Z=make_reducer('tsne', n_components).fit_transform(X)
            Z_target=Z[Wref.size//F:]              # slice target portion
            return Z_target.reshape(P,E,-1)
        elif fit_space=='reference':
            print("[TSNE] 'reference' not supported; using fit_space='target'.")
        Z=make_reducer('tsne', n_components).fit_transform(W.reshape(-1,F))
        return Z.reshape(P,E,-1)

    if fit_space=='reference':
        Wref,_=flatten4(ref_weights); Xref=Wref.reshape(-1,F)
        R=make_reducer(Rname, n_components); R.fit(Xref)
        Z=R.transform(W.reshape(-1,F))
    elif fit_space=='joint':
        Wref,_=flatten4(ref_weights)
        X=np.vstack([Wref.reshape(-1,F), W.reshape(-1,F)])
        R=make_reducer(Rname, n_components); R.fit(X)
        Z=R.transform(W.reshape(-1,F))
    else: # 'target'
        R=make_reducer(Rname, n_components); R.fit(W.reshape(-1,F))
        Z=R.transform(W.reshape(-1,F))
    return Z.reshape(P,E,-1)

def _colors(P):
    return (plt.cm.tab20(np.linspace(0,1,min(P,20))) if P<=20
            else plt.cm.hsv(np.linspace(0,1,P,endpoint=False)))

def _legend(ax_or_fig, colors, labels, where='best', max_items=16, in3d=False):
    k=min(len(labels), max_items)
    proxies=[plt.Line2D([0],[0], marker='o', color='w',
                        markerfacecolor=colors[i%len(colors)], markersize=8,
                        label=labels[i]) for i in range(k)]
    if in3d:
        ax_or_fig.legend(proxies, [labels[i] for i in range(k)],
                         title="Paths (first {})".format(k),
                         fontsize=9, title_fontsize=10, loc=where, ncol=2, framealpha=0.85)
    else:
        plt.legend(proxies, [labels[i] for i in range(k)],
                   title="Paths (first {})".format(k),
                   fontsize=9, title_fontsize=10, loc=where, ncol=2, framealpha=0.85)

def plot_2d(scores, labels, alphas, title):
    P,E,D=scores.shape; assert D>=2
    cs=_colors(P); plt.figure(figsize=(9,7))
    for i in range(P):
        c=cs[i%len(cs)]; path=scores[i]
        plt.plot(path[:,0], path[:,1], color=c, alpha=0.25, lw=1.4)
        rgba=np.tile(c,(E,1)); rgba[:,3]=alphas[i]; plt.scatter(path[:,0], path[:,1], c=rgba, s=20, edgecolors='none')
    plt.title(title); plt.xlabel("Comp 1"); plt.ylabel("Comp 2"); plt.grid(alpha=0.3)
    _legend(plt.gca(), cs, labels, where='best', max_items=16, in3d=False)
    plt.tight_layout(); plt.show()

def plot_3d(scores, labels, alphas, title, elev=30, azim=45):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    P,E,D=scores.shape; assert D>=3
    cs=_colors(P); fig=plt.figure(figsize=(10,8)); ax=fig.add_subplot(111, projection='3d')
    for i in range(P):
        c=cs[i%len(cs)][:3]; path=scores[i]
        ax.plot(path[:,0], path[:,1], path[:,2], color=c, alpha=0.25, lw=1.4)
        for j in range(E):
            ax.scatter(path[j,0], path[j,1], path[j,2], color=(*c,float(alphas[i,j])), s=20, edgecolors='none')
    ax.set_title(title); ax.set_xlabel("Comp 1"); ax.set_ylabel("Comp 2"); ax.set_zlabel("Comp 3")
    ax.view_init(elev=elev, azim=azim); ax.grid(True, alpha=0.3)
    _legend(ax, cs, labels, where='best', max_items=16, in3d=True)
    plt.tight_layout(); plt.show()

def run(agent_mode='reservoir', reducer='pca', dims='both',
        rounds=1, episodes=600, time_steps=30, n_components=3, fit_space='reference', title_prefix=""):
    rewards, weights, ref_weights = get_data(agent_mode, rounds, episodes, time_steps)
    labels=[f"Angle {i*22.5:.1f}" for _ in range(rounds) for i in range(16)]
    alphas=alphas_from_rewards(rewards)

    scores=fit_transform(weights, ref_weights, reducer=reducer, n_components=n_components, fit_space=fit_space)
    name=f"{reducer.upper()} • {agent_mode.capitalize()} • fit={fit_space}"

    if dims in ('2d','both'):  plot_2d(scores[:,:,:2], labels, alphas, f"{title_prefix}{name} (2D)")
    if dims in ('3d','both') and scores.shape[2]>=3: plot_3d(scores[:,:,:3], labels, alphas, f"{title_prefix}{name} (3D)")

if __name__ == "__main__":
    # Examples:
    # run(agent_mode='normal',    reducer='pca',    dims='both', fit_space='reference')
    # run(agent_mode='reservoir', reducer='isomap', dims='2d',   fit_space='reference')
    # run(agent_mode='reservoir', reducer='tsne',   dims='both', fit_space='joint', n_components=3)
    run(agent_mode='normal', reducer='pca', dims='2d', n_components=2, fit_space='reference')
