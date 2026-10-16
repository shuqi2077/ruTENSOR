use crate::{BinaryOp, ComputeType, Error, ReductionOp, Result, UnaryOp};
use crate::operation::Kind;
use half::{bf16, f16};
use ruda_core::tensor::{DType, TensorMetadata};
use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruda_kernel::library::FastDivmod;
use ruda_kernel::tensor::RudaTensor;

#[derive(RudaLaunch, RudaType)]
pub(crate) struct Operands {
    f16_values: Sequence<Tensor<f16>>,
    bf16_values: Sequence<Tensor<bf16>>,
    f32_values: Sequence<Tensor<f32>>,
    f64_values: Sequence<Tensor<f64>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Config {
    pub kind: Kind,
    pub unary: Vec<UnaryOp>,
    pub terms: usize,
    pub addend: bool,
    pub storage: Vec<DType>,
    pub slots: Vec<usize>,
}

#[ruda]
fn unary<F: Float>(x: F, #[comptime] op: UnaryOp) -> F {
    match op {
        UnaryOp::Identity | UnaryOp::Conjugate => x,
        UnaryOp::Negate => -x,
        UnaryOp::Abs => x.abs(),
        UnaryOp::Sqrt => x.sqrt(),
        UnaryOp::Exp => x.exp(),
        UnaryOp::Log => x.ln(),
        UnaryOp::Sin => x.sin(),
        UnaryOp::Cos => x.cos(),
        UnaryOp::Tanh => x.tanh(),
        UnaryOp::Relu => select(x != x, x, select(x > F::new(0.0), x, F::new(0.0))),
        UnaryOp::Reciprocal => F::new(1.0) / x,
    }
}

#[ruda]
fn binary<F: Float>(a: F, b: F, #[comptime] op: BinaryOp) -> F {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Mul => a * b,
        BinaryOp::Min => {
            let value = select(a < b, a, b);
            // Select -0 for min(+0, -0) independently of operand order.
            let zero = select(F::new(1.0) / a < F::new(0.0), a, b);
            let value = select(a == F::new(0.0) && b == F::new(0.0), zero, value);
            select(a != a, a, select(b != b, b, value))
        }
        BinaryOp::Max => {
            let value = select(a > b, a, b);
            let zero = select(F::new(1.0) / a > F::new(0.0), a, b);
            let value = select(a == F::new(0.0) && b == F::new(0.0), zero, value);
            select(a != a, a, select(b != b, b, value))
        }
    }
}

#[ruda]
fn identity<F: Float>(#[comptime] op: ReductionOp) -> F {
    match op {
        ReductionOp::Sum => F::new(0.0),
        ReductionOp::Product => F::new(1.0),
        ReductionOp::Min => F::new(f32::INFINITY),
        ReductionOp::Max => F::new(f32::NEG_INFINITY),
    }
}

#[ruda]
fn fold<F: Float>(a: F, b: F, #[comptime] op: ReductionOp) -> F {
    match op {
        ReductionOp::Sum => a + b,
        ReductionOp::Product => a * b,
        ReductionOp::Min => binary::<F>(a, b, BinaryOp::Min),
        ReductionOp::Max => binary::<F>(a, b, BinaryOp::Max),
    }
}

#[ruda]
fn read<F: Float>(
    inputs: &Operands, strides: &Sequence<usize>,
    output_shape: &Sequence<FastDivmod<usize>>,
    reduction_shape: &Sequence<FastDivmod<usize>>,
    output_index: usize, reduction_index: usize,
    #[comptime] operand: usize, #[comptime] config: Config,
) -> F {
    let output_rank = comptime![output_shape.len()];
    let rank = comptime![output_rank + reduction_shape.len()];
    let mut position = output_index;
    let mut offset = 0usize;
    #[unroll]
    for axis in 0..output_rank {
        let axis = comptime![output_rank - axis - 1];
        let (rest, coordinate) = output_shape[axis].div_mod(position);
        position = rest;
        offset += coordinate * strides[operand * rank + axis];
    }
    position = reduction_index;
    #[unroll]
    for axis in 0..reduction_shape.len() {
        let axis = comptime![reduction_shape.len() - axis - 1];
        let (rest, coordinate) = reduction_shape[axis].div_mod(position);
        position = rest;
        offset += coordinate * strides[operand * rank + output_rank + axis];
    }
    let slot = comptime![config.slots[operand]];
    let value = match comptime![config.storage[operand]] {
        DType::F16 => F::cast_from(inputs.f16_values.index(slot)[offset]),
        DType::BF16 => F::cast_from(inputs.bf16_values.index(slot)[offset]),
        DType::F32 => F::cast_from(inputs.f32_values.index(slot)[offset]),
        DType::F64 => F::cast_from(inputs.f64_values.index(slot)[offset]),
        _ => unreachable!("validated operand storage"),
    };
    unary::<F>(value, comptime![config.unary[operand]])
}

#[ruda]
fn output_offset(
    output_index: usize, shape: &Sequence<FastDivmod<usize>>, strides: &Sequence<usize>,
) -> usize {
    let mut position = output_index;
    let mut offset = 0usize;
    #[unroll]
    for axis in 0..shape.len() {
        let axis = comptime![shape.len() - axis - 1];
        let (rest, coordinate) = shape[axis].div_mod(position);
        position = rest;
        offset += coordinate * strides[axis];
    }
    offset
}

#[ruda]
fn product_term<F: Float>(
    inputs: &Operands, strides: &Sequence<usize>,
    output_shape: &Sequence<FastDivmod<usize>>, reduction_shape: &Sequence<FastDivmod<usize>>,
    output_index: usize, reduction_index: usize, #[comptime] config: Config,
) -> F {
    let mut value = read::<F>(inputs, strides, output_shape, reduction_shape,
        output_index, reduction_index, 0usize, config.clone());
    #[unroll]
    for operand in 1..config.terms {
        value *= read::<F>(inputs, strides, output_shape, reduction_shape,
            output_index, reduction_index, operand, config.clone());
    }
    value
}

#[ruda]
fn epilogue<F: Float>(
    value: F, inputs: &Operands, strides: &Sequence<usize>,
    output_shape: &Sequence<FastDivmod<usize>>, reduction_shape: &Sequence<FastDivmod<usize>>,
    scalars: &Sequence<InputScalar>, output_index: usize, #[comptime] config: Config,
) -> F {
    let mut result = scalars[0].get::<F>() * value;
    if comptime![config.addend] {
        let beta = scalars[1].get::<F>();
        // A zero beta does not read C, including when C contains NaNs.
        if beta != F::new(0.0) {
            let addend = read::<F>(inputs, strides, output_shape,
                reduction_shape, output_index, 0, config.terms,
                config.clone());
            result += beta * addend;
        }
    }
    result
}

#[ruda(launch, address_type = "dynamic")]
fn pointwise<F: Float, O: Float>(
    inputs: Operands, output: &mut Tensor<O>,
    strides: Sequence<usize>, output_strides: Sequence<usize>,
    output_shape: Sequence<FastDivmod<usize>>, reduction_shape: Sequence<FastDivmod<usize>>,
    scalars: Sequence<InputScalar>, count: usize,
    #[comptime] config: Config, #[define(O)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS;
    if index >= count { terminate!(); }
    let mut value = F::new(0.0);
    match config.kind {
        Kind::Elementwise(op_ab, op_abc) => {
            let a = scalars[0].get::<F>() * read::<F>(&inputs, &strides,
                &output_shape, &reduction_shape, index, 0, 0usize, config.clone());
            let b = scalars[1].get::<F>() * read::<F>(&inputs, &strides,
                &output_shape, &reduction_shape, index, 0, 1usize, config.clone());
            value = binary::<F>(a, b, op_ab);
            if comptime![config.terms == 3] {
                let c = scalars[2].get::<F>() * read::<F>(&inputs, &strides,
                    &output_shape, &reduction_shape, index, 0, 2usize, config.clone());
                value = binary::<F>(value, c, op_abc);
            }
        }
        _ => {
            value = product_term::<F>(&inputs, &strides, &output_shape, &reduction_shape,
                index, 0, config.clone());
            value = epilogue::<F>(value, &inputs, &strides, &output_shape, &reduction_shape,
                &scalars, index, config);
        }
    }
    output[output_offset(index, &output_shape, &output_strides)] = O::cast_from(value);
}

#[ruda(launch, address_type = "dynamic")]
fn aggregate<F: Float, O: Float>(
    inputs: Operands, output: &mut Tensor<O>,
    strides: Sequence<usize>, output_strides: Sequence<usize>,
    output_shape: Sequence<FastDivmod<usize>>, reduction_shape: Sequence<FastDivmod<usize>>,
    scalars: Sequence<InputScalar>, count: usize, reduction_count: usize,
    #[comptime] threads: usize, #[comptime] reduction: ReductionOp,
    #[comptime] config: Config, #[define(O)] _dtype: StorageType,
) {
    let index = ABSOLUTE_POS / threads;
    if index >= count { terminate!(); }
    let lane = UNIT_POS as usize;
    let mut value = identity::<F>(reduction);
    let mut position = lane;
    while position < reduction_count {
        let next = product_term::<F>(&inputs, &strides, &output_shape, &reduction_shape,
            index, position, config.clone());
        value = fold::<F>(value, next, reduction);
        if reduction_count - position <= threads { break; }
        position += threads;
    }
    let mut shared = SharedMemory::<F>::new(threads);
    shared[lane] = value;
    let mut step = RUDA_DIM as usize / 2;
    while step > 0 {
        sync_ruda();
        if lane < step {
            shared[lane] = fold::<F>(shared[lane], shared[lane + step], reduction);
        }
        step /= 2;
    }
    if lane == 0 {
        value = epilogue::<F>(shared[0], &inputs, &strides, &output_shape, &reduction_shape,
            &scalars, index, config);
        output[output_offset(index, &output_shape, &output_strides)] = O::cast_from(value);
    }
}

fn grid<R: Runtime>(client: &ComputeClient<R>, units: usize, dim: RudaDim) -> Result<(RudaCount, AddressType)> {
    let threads = dim.num_elems() as usize;
    let groups = units.div_ceil(threads);
    let limits = client.properties().hardware.max_ruda_count;
    if limits.0 == 0 || limits.1 == 0 || limits.2 == 0 {
        return Err(Error::UnsupportedDevice("empty dispatch grid".into()));
    }
    let x = groups.min(limits.0 as usize).max(1);
    let remaining = groups.div_ceil(x);
    let y = remaining.min(limits.1 as usize).max(1);
    let z = remaining.div_ceil(y).max(1);
    if z > limits.2 as usize {
        return Err(Error::UnsupportedDevice("operation exceeds the device dispatch grid".into()));
    }
    let launched = x.checked_mul(y).and_then(|n| n.checked_mul(z))
        .and_then(|n| n.checked_mul(threads)).ok_or(Error::Overflow)?;
    Ok((RudaCount::Static(x as u32, y as u32, z as u32), AddressType::from_len(launched)))
}

pub(crate) struct Launch<'a, R: Runtime> {
    pub inputs: &'a [RudaTensor<R>],
    pub output: &'a RudaTensor<R>,
    pub strides: Vec<usize>,
    pub reduction_extents: &'a [usize],
    pub scalars: &'a [f64],
    pub count: usize,
    pub reduction_count: usize,
    pub threads: usize,
    pub compute: ComputeType,
    pub config: Config,
}

impl<R: Runtime> Launch<'_, R> {
    pub fn submit(self) -> Result<()> {
        match self.compute {
            ComputeType::F32 => self.submit_typed::<f32>(),
            ComputeType::F64 => self.submit_typed::<f64>(),
        }
    }

    fn submit_typed<F: Float>(self) -> Result<()> {
        let client = &self.output.client;
        let inputs = OperandsLaunch::new(
            self.inputs.iter().filter(|x| x.dtype == DType::F16)
                .map(|x| x.clone().into_tensor_arg()).collect::<SequenceArg<R, Tensor<f16>>>(),
            self.inputs.iter().filter(|x| x.dtype == DType::BF16)
                .map(|x| x.clone().into_tensor_arg()).collect::<SequenceArg<R, Tensor<bf16>>>(),
            self.inputs.iter().filter(|x| x.dtype == DType::F32)
                .map(|x| x.clone().into_tensor_arg()).collect::<SequenceArg<R, Tensor<f32>>>(),
            self.inputs.iter().filter(|x| x.dtype == DType::F64)
                .map(|x| x.clone().into_tensor_arg()).collect::<SequenceArg<R, Tensor<f64>>>(),
        );
        let strides = self.strides.into_iter().collect::<SequenceArg<R, usize>>();
        let output_strides = self.output.meta.strides().iter().copied().collect::<SequenceArg<R, usize>>();
        let output_shape = self.output.meta.shape().iter().map(|&d| d.max(1))
            .collect::<SequenceArg<R, FastDivmod<usize>>>();
        let reduction_shape = self.reduction_extents.iter().map(|&d| d.max(1))
            .collect::<SequenceArg<R, FastDivmod<usize>>>();
        let scalars = self.scalars.iter().map(|&x| InputScalar::new(x, self.compute.dtype()))
            .collect::<SequenceArg<R, InputScalar>>();
        let address = self.inputs.iter().fold(self.output.required_address_type(),
            |address, input| address.max(input.required_address_type()))
            .max(AddressType::from_len(self.count.max(self.reduction_count)))
            .max(AddressType::from_len(self.count * self.threads));
        if self.reduction_extents.is_empty() {
            let dim = RudaDim::new(client.properties(), self.count);
            let (ruda_count, grid_address) = grid(client, self.count, dim)?;
            pointwise::launch::<F, R>(client, ruda_count, dim,
                address.max(grid_address), inputs, self.output.clone().into_tensor_arg(), strides, output_strides,
                output_shape, reduction_shape, scalars, self.count, self.config, self.output.dtype.into());
        } else {
            let dim = RudaDim::new_1d(self.threads as u32);
            let reduction = match self.config.kind {
                Kind::Reduction(op) => op,
                _ => ReductionOp::Sum,
            };
            let (ruda_count, grid_address) = grid(client, self.count * self.threads, dim)?;
            aggregate::launch::<F, R>(client, ruda_count, dim,
                address.max(grid_address), inputs, self.output.clone().into_tensor_arg(), strides, output_strides,
                output_shape, reduction_shape, scalars, self.count, self.reduction_count,
                self.threads, reduction, self.config, self.output.dtype.into());
        }
        Ok(())
    }
}
