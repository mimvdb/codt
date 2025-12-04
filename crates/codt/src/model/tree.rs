use std::{fmt::Display, sync::Arc};

use crate::tasks::OptimizationTask;

pub enum Tree<OT: OptimizationTask> {
    Branch(BranchNode<OT>),
    Leaf(LeafNode<OT>),
}

impl<OT: OptimizationTask> Display for Tree<OT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Branch(branch) => write!(
                f,
                "[{}<={}:{},{}]",
                branch.split_feature, branch.split_threshold, branch.left_child, branch.right_child
            ),
            Self::Leaf(leaf) => write!(f, "{}", leaf.label),
        }
    }
}

impl<OT: OptimizationTask> Tree<OT> {
    pub fn cost(&self) -> OT::CostType {
        match self {
            Self::Branch(branch) => branch.cost,
            Self::Leaf(leaf) => leaf.cost,
        }
    }

    pub fn predict(&self, features: Vec<f64>) -> OT::LabelType {
        match self {
            Self::Branch(branch) => {
                if features[branch.split_feature] <= branch.split_threshold {
                    branch.left_child.predict(features)
                } else {
                    branch.right_child.predict(features)
                }
            }
            Self::Leaf(leaf) => leaf.label,
        }
    }
}

pub struct BranchNode<OT: OptimizationTask> {
    pub cost: OT::CostType,
    pub split_feature: usize,
    pub split_threshold: f64,
    pub left_child: Arc<Tree<OT>>,
    pub right_child: Arc<Tree<OT>>,
}

pub struct LeafNode<OT: OptimizationTask> {
    pub cost: OT::CostType,
    pub label: OT::LabelType,
}

impl<OT: OptimizationTask> Clone for LeafNode<OT> {
    fn clone(&self) -> Self {
        Self {
            cost: self.cost,
            label: self.label,
        }
    }
}
