use std::collections::{BTreeMap, BTreeSet};
use crate::{Error, Mode, OperandDescriptor, OperationDescriptor, Plan, Result, TensorDescriptor};
use crate::plan::merge_extent;
use ruda_kernel::{dsl::Runtime, tensor::RudaTensor};

#[derive(Clone, Debug)]
enum Tree {
    Input(usize),
    Pair(Box<Tree>, Box<Tree>, Vec<Mode>),
}

#[derive(Clone)]
struct Candidate {
    products: u128,
    tree: Tree,
}

#[derive(Clone, Debug)]
enum Node {
    Input(usize),
    Pair { left: Box<Node>, right: Box<Node>, plan: Plan },
}

#[derive(Clone, Debug)]
pub(crate) struct ContractionPath {
    root: Node,
    pub products: u128,
}

struct Search<'a> {
    operation: &'a OperationDescriptor,
    extents: BTreeMap<Mode, usize>,
    memo: BTreeMap<Vec<usize>, Candidate>,
}

impl Search<'_> {
    fn modes(&self, subset: &[usize]) -> Vec<Mode> {
        if subset.len() == 1 {
            return self.operation.inputs[subset[0]].modes.clone();
        }
        if subset.len() == self.operation.terms {
            return self.operation.output_modes.clone();
        }
        let present: BTreeSet<_> = subset.iter()
            .flat_map(|&i| self.operation.inputs[i].modes.iter().copied()).collect();
        let mut needed: BTreeSet<_> = self.operation.output_modes.iter().copied().collect();
        for (i, input) in self.operation.inputs.iter().enumerate() {
            if !subset.contains(&i) { needed.extend(input.modes.iter().copied()); }
        }
        present.intersection(&needed).copied().collect()
    }

    fn solve(&mut self, subset: &[usize]) -> Result<Candidate> {
        if let Some(candidate) = self.memo.get(subset) { return Ok(candidate.clone()); }
        if subset.len() == 1 {
            return Ok(Candidate { products: 0, tree: Tree::Input(subset[0]) });
        }
        let mut best = None;
        self.partitions(subset, 1, &mut vec![subset[0]], &mut Vec::new(), &mut best)?;
        let candidate = best.ok_or(Error::Overflow)?;
        self.memo.insert(subset.to_vec(), candidate.clone());
        Ok(candidate)
    }

    fn partitions(
        &mut self, subset: &[usize], position: usize,
        left: &mut Vec<usize>, right: &mut Vec<usize>, best: &mut Option<Candidate>,
    ) -> Result<()> {
        if position != subset.len() {
            left.push(subset[position]);
            self.partitions(subset, position + 1, left, right, best)?;
            left.pop();
            right.push(subset[position]);
            self.partitions(subset, position + 1, left, right, best)?;
            right.pop();
            return Ok(());
        }
        if right.is_empty() { return Ok(()); }
        let a = match self.solve(left) {
            Ok(value) => value,
            Err(Error::Overflow) => return Ok(()),
            Err(error) => return Err(error),
        };
        let b = match self.solve(right) {
            Ok(value) => value,
            Err(Error::Overflow) => return Ok(()),
            Err(error) => return Err(error),
        };
        let axes: BTreeSet<_> = self.modes(left).into_iter().chain(self.modes(right)).collect();
        let work = if axes.iter().any(|mode| self.extents[mode] == 0) { 0 } else {
            let Some(work) = axes.iter().try_fold(1u128, |size, mode|
                size.checked_mul(self.extents[mode] as u128)) else { return Ok(()); };
            work
        };
        let Some(products) = a.products.checked_add(b.products)
            .and_then(|sum| sum.checked_add(work)) else { return Ok(()); };
        if best.as_ref().is_none_or(|previous| products < previous.products) {
            *best = Some(Candidate { products,
                tree: Tree::Pair(Box::new(a.tree), Box::new(b.tree), self.modes(subset)) });
        }
        Ok(())
    }
}

impl ContractionPath {
    pub fn new(operation: &OperationDescriptor) -> Result<Self> {
        let mut extents = BTreeMap::new();
        for input in &operation.inputs {
            for (&mode, &extent) in input.modes.iter().zip(&input.tensor.extents) {
                merge_extent(&mut extents, mode, extent)?;
            }
        }
        let mut search = Search { operation, extents, memo: BTreeMap::new() };
        let candidate = search.solve(&(0..operation.terms).collect::<Vec<_>>())?;
        let (root, _) = Self::lower(candidate.tree, operation, &search.extents, true)?;
        Ok(Self { root, products: candidate.products })
    }

    fn lower(
        tree: Tree, operation: &OperationDescriptor, extents: &BTreeMap<Mode, usize>, root: bool,
    ) -> Result<(Node, OperandDescriptor)> {
        match tree {
            Tree::Input(index) => Ok((Node::Input(index), operation.inputs[index].clone())),
            Tree::Pair(left, right, modes) => {
                let (left, a) = Self::lower(*left, operation, extents, false)?;
                let (right, b) = Self::lower(*right, operation, extents, false)?;
                let output = if root { operation.output.clone() } else {
                    TensorDescriptor::contiguous(
                        &modes.iter().map(|mode| extents[mode]).collect::<Vec<_>>(),
                        operation.compute.dtype(),
                    )?
                };
                let result = OperandDescriptor::new(output.clone(), &modes)?;
                let plan = Plan::new(OperationDescriptor::contraction(
                    a, b, None, output, &modes, operation.compute)?)?;
                Ok((Node::Pair { left: Box::new(left), right: Box::new(right), plan }, result))
            }
        }
    }

    pub fn execute<R: Runtime>(&self, inputs: &[&RudaTensor<R>]) -> Result<RudaTensor<R>> {
        self.root.execute(inputs)
    }

    pub fn execute_into<R: Runtime>(
        &self, inputs: &[&RudaTensor<R>], output: RudaTensor<R>,
    ) -> Result<RudaTensor<R>> {
        match &self.root {
            Node::Pair { left, right, plan } => {
                let a = left.execute(inputs)?;
                let b = right.execute(inputs)?;
                plan.execute_into(&[&a, &b], output, &[1.0])
            }
            Node::Input(_) => unreachable!("optimized paths have at least two operands"),
        }
    }
}

impl Node {
    fn execute<R: Runtime>(&self, inputs: &[&RudaTensor<R>]) -> Result<RudaTensor<R>> {
        match self {
            Self::Input(index) => Ok((*inputs[*index]).clone()),
            Self::Pair { left, right, plan } => {
                let a = left.execute(inputs)?;
                let b = right.execute(inputs)?;
                plan.execute(&[&a, &b], &[1.0])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ComputeType, EinsumPlan};
    use ruda_core::tensor::DType;

    #[test]
    fn matrix_chain_chooses_minimum_product_path() {
        let inputs = [[10, 100], [100, 5], [5, 50]].map(|shape|
            TensorDescriptor::contiguous(&shape, DType::F32).unwrap());
        let plan = EinsumPlan::new("ab,bc,cd->ad", &inputs).unwrap();
        assert_eq!(plan.planned_scalar_products(), None);
        let optimized = plan.optimize().unwrap();
        assert_eq!(optimized.planned_scalar_products(), Some(7_500));
        assert_eq!(optimized.output().extents(), &[10, 50]);
    }

    #[test]
    fn intermediates_use_compute_dtype_and_final_output_keeps_storage_dtype() {
        let inputs = [([10, 100], DType::F16), ([100, 5], DType::BF16), ([5, 50], DType::F32)]
            .map(|(shape, dtype)| TensorDescriptor::contiguous(&shape, dtype).unwrap());
        let einsum = EinsumPlan::with_options("ab,bc,cd->ad", &inputs, DType::F16, ComputeType::F32).unwrap();
        let path = ContractionPath::new(einsum.plan().operation()).unwrap();
        let Node::Pair { left, right, plan } = path.root else { panic!("missing final contraction") };
        assert_eq!(plan.operation().output().dtype(), DType::F16);
        let Node::Pair { plan, .. } = *left else { panic!("expected A @ B intermediate") };
        assert_eq!(plan.operation().output().dtype(), DType::F32);
        assert!(matches!(*right, Node::Input(2)));
    }

    #[test]
    fn unary_einsum_does_not_create_binary_path() {
        let input = TensorDescriptor::contiguous(&[3, 3], DType::F64).unwrap();
        let plan = EinsumPlan::new("ii->", &[input]).unwrap().optimize().unwrap();
        assert_eq!(plan.planned_scalar_products(), None);
        assert!(plan.output().extents().is_empty());
    }
}
