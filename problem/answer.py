import sys
from typing import Any

import numpy as np
from openfermion.transforms import jordan_wigner
from openfermion.utils import load_operator
import statistics

from quri_parts.algo.ansatz import HardwareEfficientReal, EntanglementPatternType, SymmetryPreserving, TwoLocal, build_entangler_map
from quri_parts.algo.optimizer import NFTfit, NFT, OptimizerStatus
from quri_parts.core.estimator.gradient import numerical_gradient_estimates, parameter_shift_gradient_estimates
from quri_parts.core.measurement import bitwise_commuting_pauli_measurement
from quri_parts.core.sampling.shots_allocator import (
    create_equipartition_shots_allocator,
)
from quri_parts.core.state import ParametricCircuitQuantumState, ComputationalBasisState
from quri_parts.openfermion.operator import operator_from_openfermion_op
from quri_parts.qiskit.circuit import circuit_from_qiskit

from quri_parts.circuit.circuit_parametric import UnboundParametricQuantumCircuit

from qiskit import QuantumCircuit
from quri_parts.circuit import CNOT
from quri_parts.circuit.gates import H
import random

sys.path.append("../")
from utils.challenge_2023 import ChallengeSampling, QuantumCircuitTimeExceededError

challenge_sampling = ChallengeSampling(noise=True)

def SU4_Ansatz(num_qubits, depth, params, optimizing = 0):
    qc = QuantumCircuit(num_qubits)
    returns = []
    parametric = UnboundParametricQuantumCircuit(num_qubits)
    iteration = 0
    flagged = False
    for layer in range(depth):
        for qubit in range(int(layer % 2 == 0), num_qubits-1, 2):
            if(not flagged and iteration != optimizing):
                qc.rx(params[iteration][0], qubit)
                qc.ry(params[iteration][1], qubit)
                qc.rz(params[iteration][2], qubit)
                qc.rx(params[iteration][3], qubit + 1)
                qc.ry(params[iteration][4], qubit + 1)
                qc.rz(params[iteration][5], qubit + 1)
            if(iteration == optimizing):
                temp = UnboundParametricQuantumCircuit(num_qubits)
                temp = temp.combine(circuit_from_qiskit(qiskit_circuit = qc).gates)
                returns.append(temp)
                qc = QuantumCircuit(num_qubits)
                parametric.add_ParametricRX_gate(qubit)
                parametric.add_ParametricRY_gate(qubit)
                parametric.add_ParametricRZ_gate(qubit)
                parametric.add_ParametricRX_gate(qubit+1)
                parametric.add_ParametricRY_gate(qubit+1)
                parametric.add_ParametricRZ_gate(qubit+1)
                returns.append(parametric)
                flagged = True
            iteration += 1
    for qubit1 in range(0, num_qubits):
        for qubit2 in range(0, int(num_qubits/2)):
            if(qubit1 != qubit2):
                qc.cnot(qubit1, qubit2)
                qc.h(qubit1)
                qc.h(qubit2)

    temp = UnboundParametricQuantumCircuit(num_qubits)
    temp = temp.combine(circuit_from_qiskit(qiskit_circuit = qc).gates)
    returns.append(temp)
    return returns

def cost_fn(hamiltonian, parametric_state, param_values, estimator):
    estimate = estimator(hamiltonian, parametric_state, [param_values])
    return estimate[0].value.real

def optimization_sweep(hw_oracle, hamiltonian, num_qubits, depth):
    iterations = 0
    for layer in range(depth):
        for qubit in range(int(layer % 2 == 0), num_qubits-1, 2):
            iterations += 1

    hardware_type = "sc"
    shots_allocator = create_equipartition_shots_allocator()
    measurement_factory = bitwise_commuting_pauli_measurement
    n_shots = 10**3

    estimator = (
        challenge_sampling.create_concurrent_parametric_sampling_estimator(
            n_shots, measurement_factory, shots_allocator, hardware_type
        )
    )

    def c_fn(param_values):
        return cost_fn(hamiltonian, parametric_state, param_values, estimator)

    def g_fn(param_values):
        grad = parameter_shift_gradient_estimates(
                hamiltonian, parametric_state, param_values, estimator
        )
        return np.asarray([i.real for i in grad.values])

    parameters_per_block = [np.random.rand(6)*2*np.pi*0.001 for i in range(iterations)]
    parameters_per_block2 = parameters_per_block
    iterationTotal = 0
    while True:
        try:
            for i in range(iterations):
                optimizer = NFT(randomize=True, reset_interval=100)
                opt_state = optimizer.get_init_state(parameters_per_block[i])
                hw_ansatz = SU4_Ansatz(num_qubits, depth, parameters_per_block, i)
                hw_hf = hw_ansatz[0]
                for item in hw_ansatz[1:-1]:
                    hw_hf = hw_hf.combine(item)
                hw_hf = hw_hf.combine(hw_oracle)
                parametric_state = ParametricCircuitQuantumState(num_qubits, hw_hf)
                prior_answer = opt_state.params
                for k in range(40):
                    opt_state = optimizer.step(opt_state, c_fn, g_fn)
                    print(f"iteration {iterationTotal+1}")
                    print(opt_state.cost)
                    prior_answer = opt_state.params
                    cost = opt_state.cost
                    iterationTotal += 1
                parameters_per_block2[i] = opt_state.params
            parameters_per_block = parameters_per_block2
        except QuantumCircuitTimeExceededError:
           print("Reached the limit of shots")
           return cost, iterationTotal

    return cost, iterationTotal

class RunAlgorithm:
    def __init__(self) -> None:
        challenge_sampling.reset()

    def result_for_evaluation(self) -> tuple[Any, float]:
        energy_final = self.get_result()
        qc_time_final = challenge_sampling.total_quantum_circuit_time

        return energy_final, qc_time_final

    def get_result(self) -> float:
        n_qubits = 8
        ham = load_operator(
            file_name=f"{n_qubits}_qubits_H",
            data_directory="../hamiltonian",
            plain_text=False,
        )
        jw_hamiltonian = jordan_wigner(ham)
        hamiltonian = operator_from_openfermion_op(jw_hamiltonian)
        hf_gates = ComputationalBasisState(n_qubits, bits=0b00001111).circuit.gates
        cost, iteration = optimization_sweep(hf_gates, hamiltonian, n_qubits, n_qubits)

        print(f"iteration used: {iteration+1}")

        return cost

if __name__ == "__main__":
    run_algorithm = RunAlgorithm()
    print(run_algorithm.get_result())
