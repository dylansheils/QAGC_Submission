# Quantum Algorithm Grand Challenge Submission

# Overview of Quantum Algorithm Grand Challenge <a id="Overview"></a>
Quantum Algorithm Grand Challenge (QAGC) is a global online contest for students, researchers, and others who learn quantum computation and quantum chemistry around the world.

From May 3 to July 31, 2023, participants will solve a problem that focus on the industrial application of the NISQ algorithms.

QAGC web-site:  https://www.qagc.org/

# Introduction <a id="Introduction"></a>

As Quantum Computing technology evolves with qubit capacity regularly duplicating, we need to understand how to make better use of Noisy Intermediate-Scale Quantum (NISQ) devices and create algorithms that will enable industrial applications. To identify how to shape the direction for promoting the NISQ algorithm for practical industrial application, it is important to clarify the evaluation criteria to compare the algorithm's performance and define the key factors to take into account. 

We hold a global online contest, the QAGC, to explore practical uses for NISQ devices, visualize bottlenecks in NISQ device utilization, and create a metric for benchmarking the NISQ algorithms.

## Background <a id="Introduction_1"></a>

The materials around us are constructed from molecules and the microscopic behavior is dominated by quantum mechanics. Quantum chemistry is widely used to understand the chemical behavior of these materials in not only academic studies but also material design in industries.

Quantum chemistry is considered one of the most promising fields for considering practical industrial applications of NISQ devices and algorithms.  However, although various NISQ algorithms have been proposed in recent years, it is still far from practical industrial applications. 

For practical industrial applications of NISQ algorithms, it is important to develop new useful NISQ algorithms and define evaluation criteria for accurately comparing the performance of various algorithms and define the key factors to take into account.

Based on these situations, the focuses of QAGC are on the industrial application and defining evaluation criteria for appropriate performance comparison of NISQ algorithms. We have prepared a simulator that reflect the features of NISQ devices and suitable model for the problem to achieve these goals. Below, we will explain each of them.

## Model description <a id="Introduction_2"></a> 

The ground state energy of a molecule is an important quantity for understanding its properties and behavior, and many quantum chemistry studies focus on the ground state energy of individual atoms or molecules. 

In QAGC, the task of participants is to calculate the ground state energy of a model (Hamiltonian) which we have prepared. From the focus of QAGC, the hamiltonian should have some properties as follows:

- Has similar properties as the molecular Hamiltonian used in quantum chemistry.

  - The number of terms of the hamiltonian is $O(N^4)$, which is the same as the molecular Hamiltonian. Then we can compare the performance of grouping methods that reduce the number of measurements.
  - This hamiltonian has the same order of operator norm as the molecular Hamiltonian. Therefore, the resulting ground state energy scale is similar to the scale in quantum chemistry.


- The exact value of ground state energy of this hamiltonian can be calculated classically for the arbitrary size of the system.
  
  - Our aim through QAGC is also to create a common metric that can evaluate various NISQ algorithms. For evaluating algorithms in large qubit systems that cannot be simulated in classical computers, it will be necessary to know the exact value of the quantity to be measured as a reference value for evaluating NISQ algorithms. 

We have prepared a hamiltonian that satisfies all of these properties. The detail of the hamiltonian and the problem statement in QAGC is written in [Problem](#problem).

## NISQ device simulation <a id="Introduction_3"></a>
To explore the practical applications of NISQ devices and visualize bottlenecks in their utilization, it is necessary to use simulators that reflect the features of NISQ devices.

In QAGC, the participants need to use a sampling simulator we have provided. This simulator automatically performs sampling that reflects the functioning of NISQ devices and calculates an expected execution time. 
When sampling within an algorithm, it is restricted to not exceed *1000s* of the expected execution time. We will explain this limitation in [Evaluation Criteria](#EvaluationCriteria). And more detail of these NISQ device simulation and the expected execution time are written in `technical_details.md`.

# Problem description <a id="problem"></a>

## Fermi-Hubbard Model <a id="problem_1"></a>
The Fermi-Hubbard model is a model used to describe the properties of strongly correlated electron systems, which are solids with strong electron correlation effects. It is used to explain important physical phenomena such as magnetism, Mott insulators, and high-temperature superconductors. 


In QAGC, we deal with a one-dimensional orbital rotated Fermi-Hubbard model with **periodic boundary conditions**. The hamiltonian of one-dimensional Fermi-Hubbard model is as follows:

$$
    H = - t \sum_{i=0}^{N-1} \sum_{\sigma=\uparrow, \downarrow} (a^\dagger_{i, \sigma}  a_{i+1, \sigma} +  a^\dagger_{i, \sigma}  a_{i+1, \sigma})  - \mu \sum_{i=0}^{N-1} \sum_{\sigma=\uparrow, \downarrow}  a^\dagger_{i, \sigma} a_{i, \sigma} + U \sum_{i=0}^{N-1} a^\dagger_{i, \uparrow}  a_{i, \uparrow}  a^\dagger_{i, \downarrow} a_{i, \downarrow},
$$

where $t$ is the tunneling amplitude, $\mu$ is the chemical potential, and $U$ is the Coulomb potential. For the case of half-filling, i.e. the number of electrons is equal to the number of sites, the exact value of the ground-state energy for this Hamiltonian can be calculated by using Bethe Ansatz method. 

This time we consider the orbital rotated one-dimensional Fermi-Hubbard model. The orbital rotation means linear transformation of the creation operator $a_i^\dagger$ and annihilation operator $a_i$ by using unitary matrices

$$
    \tilde a_i^\dagger = \sum_{k=0}^{N-1} u_{ik} a_k^\dagger, \quad 
    \tilde a_i = \sum_{k=0}^{N-1} u_{ik}^* a_k.
$$

By performing orbital rotation in this way, without changing the energy eigenvalues, we can increase the number of terms to $O(N^4)$ which is the same as the molecular Hamiltonian. 

After performing orbital rotation, the Hartree-Fock calculation can be performed similar to the molecular Hamiltonian. The resulting Hartree-Fock state become:

$$
    |HF\rangle = |00001111\rangle
$$

where electrons are filled from the bottom up for a number of sites.

## Problem statement <a id="problem_2"></a>

Find the energy of the ground state of the one-dimensional orbital rotated Fermi-Hubbard model.

$$
    H = - t \sum_{i=0}^{N-1} \sum_{\sigma=\uparrow, \downarrow} (\tilde a^\dagger_{i, \sigma} \tilde a_{i+1, \sigma} + \tilde a^\dagger_{i, \sigma} \tilde a_{i+1, \sigma})  - \mu \sum_{i=0}^{N-1} \sum_{\sigma=\uparrow, \downarrow}  a^\dagger_{i, \sigma} a_{i, \sigma} + U \sum_{i=0}^{N-1} \tilde a^\dagger_{i, \uparrow} \tilde a_{i, \uparrow} \tilde a^\dagger_{i, \downarrow} \tilde a_{i, \downarrow} 
$$

The value of each parameter is $N = 4,\ t=1, \mu=1.5,\ U=3$. For QAGC, we prepared an orbital rotated Hamiltonian with the random unitary matrix $u$ and performed Hartree-Fock calculation. Hamiltonians for 4 and 8 qubits are provided in the `hamiltonian` folder in `.data` format.


# Evaluation Criteria <a id="EvaluationCriteria"></a>

First, the submitted answers are checked for compliance with the prohibited items. Then, we calculates the score based on the answers, and the ranking is determined. 

## Score

The score $S$ is calculated as the inverse of the average precision of 3 runs of the algorithm rounded to the nearest $10^{-8}$ using the following evaluation formula. 

$$
    S = \frac{1}{e}
$$

Here $e$ is the average precision.

$$
    e = \frac{1}{3}\sum_{i=1}^{3}e_i
$$

$e_i$ is the precision of the output result of the $i$ th algorithm and is defined by the following equation
using the output result of the $i$ th algorithm $E_i$ and the exact value of the Hamiltonian ground state $E_{exact}$.

$$
    e_i = |E_i - E_{exact}|
$$

## Limitation by Expected Execution Time

Reducing execution time is crucial for considering the industrial application of NISQ algorithms. Additionally, the available time to use real NISQ devices is limited. To reflect this, participants will be imposed a limit based on the expected execution time obtained from the input circuit and the number of shots. The definition of the expected execution time is explained in `technical_details.md`. 

For QAGC, sampling is restricted to ensure that the expected execution time does not exceed *1000s*.

# Implementation <a id="Implementation"></a>

Here, we will explain the necessary items for participants to implement their answer code.

Participants need to write their algorithms in `answer.py`.
- Participants should write their code in `get_result()` in `RunAlgorithm`. 
- It is also possible to add functions outside of RunAlgorithm as needed.
- The only codes that participants can modify are those in the problem folder. Do not modify the codes in the utils folder.

We have prepared an answer example in `example.py`, so please refer to it. 

Below, we will explain the sampling function and how to use the Hamiltonian of the problem.

-  ## Sampling Function

    In QAGC, all participants need to use the sampling function we have provided. Please refer to the `sampling.ipynb` in the `tutorials` folder for instructions on how to use it.

    This sampling function has the following properties:
    - Transpile the input circuit to the gates implemented on the real device for both superconducting and ion trap types and add the equivalent noise to measure it.
    - Calculate the expected execution time automatically.
    - When the expected execution time limit is reached, the error **QuantumCircuitTimeExceededError** will be output.

The details of this transpile, noise and the expected execution time are written in `technical_details.md`.
-  ## Hamiltonian

    The Hamiltonian to be used in the problem is stored in the `hamiltonian` folder in `.data` format. To load it, use `openfermion.utils.load_operator()` as follows:
    ``` python
    from openfermion.utils import load_operator

    ham = load_operator(
            file_name= 8_qubits_H", data_directory="../hamiltonian", plain_text=False
        )
    ```
    In addition to the 8-qubit Hamiltonian used for the problem, there are also 4 and 8-qubit Hamiltonians in this folder that can be freely used to verify the implemented algorithm. 

Participants can calculate the score by running `evaluator.py`.
  - **num_exec**: The number of times the algorithm is executed during evaluation.
  - **ref_value**: The reference value (exact value of the ground state energy) for each hamiltonian is listed. The score is evaluated based on this value.

Since we are dealing with a large qubits system such as 8 qubits, running evaluator.py using the code in example.py takes *6-7* hours for a single execution.

## Version

The version of the main package used in the challenge for participants will be fixed as follows:

```
quri-parts == 0.10
qiskit == 0.39.5
cirq == 1.1.0
openfermion == 1.5.1
qulacs == 0.5.6
numpy == 1.23.5
```

## Notes on Evaluation <a id="forbidden_1"></a>

The validity of the final answer will be judged by the judge based on whether it falls under the prohibited answers below. If it is deemed valid, a score will be calculated. The final decision on the validity of the answer and the score will be made by the operator.

# Copyright <a id="Copyright"></a>

The copyright of the programming code included in the participant's answer belongs to the participant.
The operator may execute and modify the programming code for the evaluation and verification of the answer, and other necessary operations of Quantum Algorithm Grand Challenge.
The operator will not publicly disclose the programming code without the permission of the submitter.
