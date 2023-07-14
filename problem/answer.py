import sys
from typing import Any
import qiskit
import numpy as np
from openfermion.transforms import jordan_wigner
from openfermion.utils import load_operator
import quri_parts
from quri_parts.qiskit.circuit import circuit_from_qiskit
from quri_parts.algo.ansatz import SymmetryPreservingReal, HardwareEfficient
from quri_parts.algo.optimizer import Adam, NFT, OptimizerStatus
from quri_parts.circuit import UnboundParametricQuantumCircuit
from quri_parts.core.estimator.gradient import parameter_shift_gradient_estimates
from quri_parts.core.measurement import bitwise_commuting_pauli_measurement
from quri_parts.core.sampling.shots_allocator import (
    create_equipartition_shots_allocator,
)

from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit
from quri_parts.circuit.circuit_parametric import UnboundParametricQuantumCircuit

from quri_parts.core.state import ParametricCircuitQuantumState, ComputationalBasisState
from quri_parts.openfermion.operator import operator_from_openfermion_op
from qiskit import QuantumCircuit
from quri_parts.circuit import CNOT
from quri_parts.circuit.gates import H
import random


sys.path.append("../")
from utils.challenge_2023 import ChallengeSampling, TimeExceededError


"""
It will take about 6-7 hours to run this code on 8 qubits.
"""

challenge_sampling = ChallengeSampling(noise=True)


def SU4_Ansatz(num_qubits, depth, params, optimizing = 0):
    qc = QuantumCircuit(num_qubits)
    returns = []
    parametric = UnboundParametricQuantumCircuit(num_qubits)
    iteration = 0
    flagged = False
    start_a = num_qubits // 2 - (1 - (num_qubits % 2))
    start_b = start_a + 1

    ranges = []
    for i in range(0, num_qubits // 2 + 2*(num_qubits % 2)):
        addition = list(range(start_a - i, start_b + i + 1))
        if(sum([element < 0 for element in addition]) == 0):
            ranges.append(addition)
    for i in range(num_qubits // 2 - 1 + 2*(num_qubits % 2), -1, -1):
        addition = list(range(start_a - i, start_b + i + 1))
        if(addition != ranges[-1] and sum([element < 0 for element in addition]) == 0):
            ranges.append(addition)

    iteration = 0
    before = UnboundParametricQuantumCircuit(num_qubits)
    after = UnboundParametricQuantumCircuit(num_qubits)
    qc = QuantumCircuit(num_qubits)
    layer_num = 0
    for layer in ranges:
        for qubit, idx in zip(layer[::2], range(0, len(layer), 2)):
            if(not flagged and iteration != optimizing and qubit >= 0 and qubit < num_qubits):
                if((layer_num + idx) % 3 == 0):
                    qc.rx(params[iteration][idx], qubit)
                if((layer_num + idx) % 3 == 1):
                    qc.rz(params[iteration][idx], qubit)
                if((layer_num + idx) % 3 == 2):
                    qc.ry(params[iteration][idx], qubit)
                qc.cnot(qubit, qubit+1)
                if((idx) % 3 == 0):
                    qc.rx(params[iteration][idx], qubit+1)
                if((idx) % 3 == 1):
                    qc.rz(params[iteration][idx], qubit+1)
                if((idx) % 3 == 2):
                    qc.ry(params[iteration][idx], qubit+1)
            if(qubit >= 0 and qubit < num_qubits): #iteration == optimizing and 
                before = before.combine(circuit_from_qiskit(qiskit_circuit = qc).gates)
                qc = QuantumCircuit(num_qubits)
                if((layer_num + idx) % 3 == 0):
                    before.add_ParametricRX_gate(qubit)
                if((layer_num + idx) % 3 == 1):
                    before.add_ParametricRZ_gate(qubit)
                if((layer_num + idx) % 3 == 2):
                    before.add_ParametricRY_gate(qubit)
                if((idx) % 3 == 0):
                    before.add_ParametricRX_gate(qubit+1)
                if((idx) % 3 == 1):
                    before.add_ParametricRZ_gate(qubit+1)
                if((idx) % 3 == 2):
                    before.add_ParametricRY_gate(qubit+1)
                flagged = True
        iteration += 1
        layer_num += 1
    before = before.combine(circuit_from_qiskit(qiskit_circuit = qc).gates)
    return before

def cost_fn(hamiltonian, parametric_state, param_values, estimator):
    estimate = estimator(hamiltonian, parametric_state, [param_values])
    return estimate[0].value.real

def optimization_sweep(hw_oracle, hamiltonian, num_qubits, depth):
    hardware_type = "it"
    shots_allocator = create_equipartition_shots_allocator()
    measurement_factory = bitwise_commuting_pauli_measurement
    n_shots = 2*10**3

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

    start_a = num_qubits // 2 - (1 - (num_qubits % 2))
    start_b = start_a + 1

    ranges = []
    for i in range(0, num_qubits // 2 + 2*(num_qubits % 2)):
        addition = list(range(start_a - i, start_b + i + 1))
        if(sum([element < 0 for element in addition]) == 0):
            ranges.append(addition)
    for i in range(num_qubits // 2 - 1 + 2*(num_qubits % 2), -1, -1):
        addition = list(range(start_a - i, start_b + i + 1))
        if(addition != ranges[-1] and sum([element < 0 for element in addition]) == 0):
            ranges.append(addition)

    parameters_per_block = []
    counter = 0
    for layer in ranges:
        counter += 2*len(layer[::2])
    parameters_per_block.append(2*np.pi*np.random.rand(counter))
    parameters_per_block2 = parameters_per_block
    iterationTotal = 0
    while True:
        try:
            #for i in range(0):
            optimizer = Adam(ftol=10e-5)
            opt_state = optimizer.get_init_state(parameters_per_block[i])
            hw_hf = SU4_Ansatz(num_qubits, depth, parameters_per_block, i)
            hw_hf = hw_hf.combine(hw_oracle)
            parametric_state = ParametricCircuitQuantumState(num_qubits, hw_hf)
            prior_answer = opt_state.params
            #for k in range(30):
            best_solution = ([], 0)
            while(True):
                opt_state = optimizer.step(opt_state, c_fn, g_fn)
                print(f"iteration {iterationTotal+1}")
                print(opt_state.cost)
                cost = opt_state.cost
                if(cost < best_solution[1]):
                    best_solution = (opt_state.params, opt_state.cost)
                iterationTotal += 1
            parameters_per_block2[i] = opt_state.params
            parameters_per_block = parameters_per_block2
        except TimeExceededError:
           print("Reached the limit of shots")
           return best_solution[1], iterationTotal

    return best_solution[1], iterationTotal

class RunAlgorithm:
    def __init__(self) -> None:
        challenge_sampling.reset()

    def result_for_evaluation(self) -> tuple[Any, float]:
        energy_final = self.get_result()
        qc_time_final = challenge_sampling.total_quantum_circuit_time

        return energy_final, qc_time_final

    def get_result(self) -> Any:
        n_site = 4
        n_qubits = 2 * n_site
        ham = load_operator(
            file_name=f"{n_qubits}_qubits_H",
            data_directory="../hamiltonian",
            plain_text=False,
        )
        jw_hamiltonian = jordan_wigner(ham)
        hamiltonian = operator_from_openfermion_op(jw_hamiltonian)

        # make hf + HEreal ansatz
        hf_gates = ComputationalBasisState(n_qubits, bits=0b00001111).circuit.gates
        hf_circuit = LinearMappedUnboundParametricQuantumCircuit(n_qubits).combine(hf_gates)

        #hw_ansatz1 = HardwareEfficient(qubit_count=n_qubits, reps=n_qubits//2)
        #hf_circuit.extend(hw_ansatz1)

        parametric_state = ParametricCircuitQuantumState(n_qubits, hf_circuit)
        hardware_type = "it"
        shots_allocator = create_equipartition_shots_allocator()
        measurement_factory = bitwise_commuting_pauli_measurement
        n_shots = 2*10**3

        sampling_estimator = (
            challenge_sampling.create_concurrent_parametric_sampling_estimator(
                n_shots, measurement_factory, shots_allocator, hardware_type
            )
        )

        cost, iteration = optimization_sweep(hf_gates, hamiltonian, n_qubits, n_qubits)
        print(f"iteration used: {iteration}")
        return cost


if __name__ == "__main__":
    run_algorithm = RunAlgorithm()
    print(run_algorithm.get_result())