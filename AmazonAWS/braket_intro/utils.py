from qutip import *
from matplotlib import pyplot, animation
from pennylane import numpy as np
from braket.jobs.metrics_data.definitions import MetricType

import pandas as pd


def plot_job_metrics(job):
    df = pd.DataFrame(job.metrics(metric_type=MetricType.ITERATION_NUMBER))
    df.sort_values(by=["iteration_number"], inplace=True)

    fig, [ax1, ax2] = pyplot.subplots(1, 2)
    fig.set_figwidth(15)
    fig.suptitle("Job metrics", size=16, y=0.99)

    ax1.plot("iteration_number", "expval", data=df)
    ax1.axhline(y=-1, color='black', linestyle='--')
    ax1.set_xlabel('iteration', fontsize=18)
    ax1.set_ylabel(r'$\langle Z\rangle$', fontsize=18, rotation=90)
    ax1.set_xlim([-1, 25])

    ax2.plot("iteration_number", "theta", data=df)
    ax2.axhline(y=np.pi, color='black', linestyle='--')
    ax2.set_xlabel('Iteration', fontsize=18)
    ax2.set_ylabel(r'$\theta$', fontsize=18, rotation=90)
    ax2.set_xlim([-1, 25])

    pyplot.show()


def bloch_sphere_animation():
    fig = pyplot.figure()
    ax = fig.add_subplot(azim=-60, elev=20, projection="3d")
    sphere = qutip.Bloch(axes=ax)
    thetas = np.linspace(0, np.pi, 20)

    def animate(theta):
        sphere.clear()
        sphere.xlabel = ['$x$', '']
        sphere.ylabel = ['$y$', '']
        sphere.zlabel = ['$z$', '']
        sphere.vector_width = 1
        sphere.vector_color = ['blue']
        sphere.add_vectors([0, np.sin(theta), np.cos(theta)])
        sphere.add_annotation(
            [0, 0.05, 1.05],
            '$|0\\rangle$',
            verticalalignment='bottom',
            horizontalalignment='left',
            color='red',
            fontsize=15
        )
        sphere.add_annotation(
            [0, 0.05, -1.1],
            '$|1\\rangle$',
            verticalalignment='bottom',
            horizontalalignment='left',
            color='red',
            fontsize=15
        )
        sphere.make_sphere()

    def init_func():
        pyplot.close()

    return animation.FuncAnimation(
        fig,
        animate,
        thetas,
        repeat=True,
        init_func=init_func
    )
