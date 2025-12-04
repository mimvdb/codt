use crate::model::dataset::DataSet;
use crate::model::dataview::FeatureValue;
use crate::model::{dataview::DataView, instance::Instance};

use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Range, Sub, SubAssign};

pub mod accuracy;
pub mod squared_error;

pub trait CostSum<LabelType, InstanceType, CostType>:
    for<'a> AddAssign<&'a Self>
    + for<'a> AddAssign<&'a InstanceType>
    + for<'a> SubAssign<&'a Self>
    + for<'a> SubAssign<&'a InstanceType>
    + Clone
    + Send
    + Sync
{
    fn label(&self) -> LabelType;
    fn cost(&self) -> CostType;
    fn clear(&mut self);
}

/// The TryInto<f64> is only required for global best first search. May be implemented using `unimplemented` macro if not required.
pub trait Cost:
    Clone
    + Copy
    + Debug
    + Display
    + Add<Output = Self>
    + Sub<Output = Self>
    + TryInto<f64, Error: Debug>
    + Send
    + Sync
{
    /// The minimum possible cost, to e.g. initialize lower bounds. Requires that ZERO_COST + ZERO_COST = ZERO_COST. For example 0 or 0.0
    const ZERO: Self;

    /// Returns true if the cost is zero. Use this for comparison to prevent floating point errors.
    fn is_zero(&self) -> bool;

    fn strictly_greater_than(&self, other: &Self) -> bool;
    fn strictly_less_than(&self, other: &Self) -> bool;

    fn greater_or_not_much_less_than(&self, other: &Self) -> bool;
    fn less_or_not_much_greater_than(&self, other: &Self) -> bool;

    /// Should only be used for sorting, e.g. in a priority queue.
    fn to_order(&self) -> impl Ord;
}

/// f64::EPSILON is too small, due to error accumulation in floating point arithmetic.
const EPSILON: f64 = 1e-7;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FloatCost(pub f64);

impl From<f64> for FloatCost {
    fn from(value: f64) -> Self {
        FloatCost(value)
    }
}

impl From<FloatCost> for f64 {
    fn from(value: FloatCost) -> Self {
        value.0
    }
}

impl Display for FloatCost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Add for FloatCost {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        FloatCost(self.0 + other.0)
    }
}

impl Sub for FloatCost {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        FloatCost(self.0 - other.0)
    }
}

impl Cost for FloatCost {
    const ZERO: Self = FloatCost(0.0);

    fn is_zero(&self) -> bool {
        self.0 < EPSILON
    }

    fn strictly_greater_than(&self, other: &Self) -> bool {
        self.0 > other.0 + EPSILON
    }

    fn strictly_less_than(&self, other: &Self) -> bool {
        self.0 < other.0 - EPSILON
    }

    fn to_order(&self) -> impl Ord {
        (self.0 / EPSILON) as i64
    }

    fn greater_or_not_much_less_than(&self, other: &Self) -> bool {
        self.0 >= other.0 - EPSILON
    }

    fn less_or_not_much_greater_than(&self, other: &Self) -> bool {
        self.0 <= other.0 + EPSILON
    }
}

pub trait OptimizationTask {
    /// The label type, e.g. a class label for classification tasks, or a regression target for regression tasks.
    type LabelType: Clone + Copy + Display + Send + Sync;
    /// The instance type. For classification and regression, each instance only has a label, see `LabeledInstance`.
    type InstanceType: Instance + Send + Sync;
    type CostType: Cost;
    /// A type from which the cost is easily derivable. When a CostSum for disjoint datasets
    /// are summed, it results in the CostSum of their union.
    type CostSumType: CostSum<Self::LabelType, Self::InstanceType, Self::CostType>;
    type ExtraDataviewData: Send + Sync;

    fn preprocess_dataset(dataset: &mut DataSet<Self::InstanceType>) {
        let _ = dataset;
    }

    /// Initialize a costsum, this should only be done once at the start.
    fn init_costsum(dataset: &DataSet<Self::InstanceType>) -> Self::CostSumType;

    fn prepare_for_data(&mut self, dataview: &mut DataView<Self>)
    where
        Self: Sized,
    {
        let _ = dataview;
    }
    fn print_cost(&mut self, cost: &Self::CostType) -> String;

    fn update_lowerbound(lb: &mut Self::CostType, candidate: &Self::CostType) {
        if candidate.strictly_greater_than(lb) {
            *lb = *candidate;
        }
    }

    fn update_upperbound(ub: &mut Self::CostType, candidate: &Self::CostType) {
        if candidate.strictly_less_than(ub) {
            *ub = *candidate;
        }
    }

    fn branching_cost(&self) -> Self::CostType;

    fn greedy_value(left_costsum: &Self::CostSumType, right_costsum: &Self::CostSumType) -> f32;

    fn branch_relaxation(&self, dataview: &DataView<Self>, max_depth: u32) -> Self::CostType
    where
        Self: Sized;

    fn init_extra_dataview_data(
        dataset: &DataSet<Self::InstanceType>,
        feature_values: &[Vec<FeatureValue>],
    ) -> Self::ExtraDataviewData;

    fn worst_cost_in_range(
        dataview: &DataView<Self>,
        feature: usize,
        range: Range<usize>,
    ) -> Self::CostType
    where
        Self: Sized;
}
