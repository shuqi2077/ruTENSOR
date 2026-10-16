use crate::{ComputeType, Error, OperationDescriptor, Result, UnaryOp};
use crate::operation::Kind;
use ruda_core::tensor::{DType, Metadata, Shape};
use ruda_kernel::{dsl::Runtime, tensor::RudaTensor};
use rublas::tensor_matmul::{MatmulStrategy, matmul};
use rublas::kernel_ir::definition::MatmulSetupError;

pub(crate) fn try_product<R: Runtime>(
    operation: &OperationDescriptor,
    inputs: &[&RudaTensor<R>],
) -> Result<Option<RudaTensor<R>>> {
    if operation.kind != Kind::Contraction || operation.terms != 2
        || operation.compute != ComputeType::F32 || operation.output_modes.len() < 2
        || inputs[0].dtype != inputs[1].dtype
        || !matches!(inputs[0].dtype, DType::F16 | DType::BF16 | DType::F32)
    {
        return Ok(None);
    }
    let a = &operation.inputs[0];
    let b = &operation.inputs[1];
    if [a, b].iter().any(|input| {
        !matches!(input.unary, UnaryOp::Identity | UnaryOp::Conjugate)
            || input.modes.iter().enumerate().any(|(i, mode)| input.modes[..i].contains(mode))
    }) {
        return Ok(None);
    }
    let rank = operation.output_modes.len();
    let m = operation.output_modes[rank - 2];
    let n = operation.output_modes[rank - 1];
    let reduced: Vec<_> = a.modes.iter().chain(&b.modes)
        .filter(|mode| !operation.output_modes.contains(mode)).copied().collect();
    if reduced.len() != 2 || reduced[0] != reduced[1]
        || !a.modes.contains(&m) || a.modes.contains(&n)
        || !b.modes.contains(&n) || b.modes.contains(&m)
    {
        return Ok(None);
    }
    let k = reduced[0];
    let ak = a.modes.iter().position(|mode| *mode == k).unwrap();
    let bk = b.modes.iter().position(|mode| *mode == k).unwrap();
    let am = a.modes.iter().position(|mode| *mode == m).unwrap();
    let bn = b.modes.iter().position(|mode| *mode == n).unwrap();
    let k_extent = a.tensor.extents[ak];
    if k_extent == 0 || b.tensor.extents[bk] != k_extent
        || a.tensor.extents[am] != operation.output.extents[rank - 2]
        || b.tensor.extents[bn] != operation.output.extents[rank - 1]
    {
        return Ok(None);
    }
    let batches = &operation.output_modes[..rank - 2];
    let batch_shape = &operation.output.extents[..rank - 2];
    let view = |index: usize, rows: usize, columns: usize, row_axis: usize, column_axis: usize| {
        let descriptor = &operation.inputs[index];
        let mut shape = batch_shape.to_vec();
        shape.extend([rows, columns]);
        let mut strides: Vec<_> = batches.iter().map(|mode| {
            descriptor.modes.iter().position(|candidate| candidate == mode)
                .filter(|&axis| descriptor.tensor.extents[axis] != 1)
                .map(|axis| descriptor.tensor.strides[axis]).unwrap_or(0)
        }).collect();
        strides.extend([descriptor.tensor.strides[row_axis], descriptor.tensor.strides[column_axis]]);
        let input = inputs[index];
        RudaTensor::new(input.client.clone(), input.handle.clone(),
            Metadata::new(Shape::from(shape), strides), input.device.clone(), input.dtype)
    };
    let lhs = view(0, operation.output.extents[rank - 2], k_extent, am, ak);
    let rhs = view(1, k_extent, operation.output.extents[rank - 1], bk, bn);
    // A rejected setup has not submitted a matmul; retain the general contraction path.
    match matmul(lhs, rhs, None, MatmulStrategy::default(), operation.compute.dtype()) {
        Ok(product) => Ok(Some(product)),
        Err(MatmulSetupError::Launch(error)) => Err(Error::Launch(format!("{error}"))),
        Err(MatmulSetupError::Unavailable(_) | MatmulSetupError::InvalidConfig(_)
            | MatmulSetupError::Vectorization(_)) => Ok(None),
    }
}
