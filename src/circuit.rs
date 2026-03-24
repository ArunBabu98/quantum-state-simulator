use crate::gates::Gate;
use crate::state_vector::StateVec;

pub struct Circuit {
    num_qubits: usize,
    operations: Vec<Gate>,
}

impl Circuit {
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            operations: Vec::new(),
        }
    }

    pub fn run(&self) -> StateVec {
        let mut state = StateVec::zero_state(self.num_qubits);

        for gate in &self.operations {
            state = gate.apply(&state);
        }

        state
    }

    // --- Gate Methods (Fluent Interface) ---

    pub fn h(&mut self, target: usize) -> &mut Self {
        let gate = Gate::lift(&Gate::hadamard(), target, self.num_qubits);
        self.operations.push(gate);
        self
    }

    pub fn x(&mut self, target: usize) -> &mut Self {
        let gate = Gate::lift(&Gate::pauli_x(), target, self.num_qubits);
        self.operations.push(gate);
        self
    }

    pub fn i(&mut self, target: usize) -> &mut Self {
        let gate = Gate::lift(&Gate::identity(), target, self.num_qubits);
        self.operations.push(gate);
        self
    }

    pub fn cnot(&mut self, control: usize, target: usize) -> &mut Self {
        let gate = Gate::cnot();
        let full_gate = if self.num_qubits > 2 {
            Gate::lift_multi(&gate, control, target, self.num_qubits)
        } else {
            gate
        };

        self.operations.push(full_gate);
        self
    }
}
