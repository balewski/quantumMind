# 2025 IonQ Workshop at ORNL QCUF

Welcome to IonQ's workshop!

## Tentative schedule

8:30-9:30
* Introduction to IonQ
* Overview of the IonQ Quantum Cloud
* Submitting jobs with Qiskit
* Other supported SDKs

9:45-10:45
* IonQ API v0.4
* Native gates and compilation
* Debiasing and sharpening

11:00-12:00
* Hybrid quantum-classical algorithms
* Hosted hybrid service
* Application spotlight: Quantum TDA


## Setup

These examples use Jupyter notebooks with a few dependencies (qiskit, qiskit-ionq, some numpy and matplotlib, optional other quantum SDKs).

Please note that the IonQ integration for Qiskit is currently not compatible with Qiskit 2.0 - we recommend Qiskit 1.4 for now.

You are welcome to use your preferred approach for setting up an environment and running the notebooks (locally with venv or conda, in the cloud with Google Colab, etc.). You can install these packages into your environment, or from the notebooks.


### Generating an API key

You will need to generate an API key on the IonQ Cloud Console. Make sure you are creating a key under the correct organization (if your account belongs to multiple organizations) and project.

Each notebook includes a place to paste in an API key. You can copy the same key into all notebooks or generate multiple keys (though they will all have the same permissions if they are connected to the same project). You can also store your key as an environment variable named `IONQ_API_KEY` instead of saving it in the notebooks. For security, you can revoke API keys from the IonQ Cloud Console at any time.


## Using the QPU

You will receive a small amount of credit for running jobs on IonQ Aria, our 25-qubit system. You are welcome to run the provided examples, try modifying these examples, or try running other jobs as you prefer. The access will remain available until you use all of your credit or until the end of the user forum on Thursday, July 24.


## More information

Most of today's content is included in [IonQ's docs](https://docs.ionq.com/) with some additional information in our [resource center](https://ionq.com/resources). Please reach out to the IonQ team this week or in the future if you have additional questions, feedback, or feature requests!
