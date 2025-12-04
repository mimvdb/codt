use std::{
    collections::HashMap,
    ops::{AddAssign, Range},
    sync::{Arc, Mutex},
};

use fenwick::index::zero_based;

use rayon::prelude::*;
use thread_local::ThreadLocal;

use crate::{
    model::{
        dataview,
        tree::{BranchNode, LeafNode, Tree},
    },
    search::{node::ExpandedQueueItem, solver_impl::SolveContext, strategy::SearchStrategy},
    tasks::{Cost, CostSum, OptimizationTask},
};

/// Conceptually performs `a[i] += delta` on the original array `a`.
/// See `fenwick::array::update`. Equivalent but with looser type constraints
fn fenwick_update<T, U>(fenwick: &mut [T], i: usize, delta: U)
where
    T: AddAssign<U>,
    U: Copy,
{
    for ii in zero_based::up(i, fenwick.len()) {
        fenwick[ii] += delta;
    }
}

/// Conceptually calculates `a[0] + ... + a[i]` on the original array `a`.
/// See `fenwick::array::prefix_sum`. Equivalent but with looser type constraints
fn fenwick_prefix_sum<T>(fenwick: &[T], i: usize, accumulator: &mut T)
where
    T: for<'a> AddAssign<&'a T>,
{
    for ii in zero_based::down(i) {
        *accumulator += &fenwick[ii];
    }
}

#[allow(clippy::too_many_arguments)]
fn update_lowest_cost<OT: OptimizationTask>(
    branching_cost: OT::CostType,
    best: &mut [D1ScoreTracker<OT>], // Current bests
    range: Range<usize>,             // The range to update
    totals: &[OT::CostSumType],      // Normal vector of total costs
    totals_left: &[OT::CostSumType], // Fenwick tree of left costs
    reverse_fenwick: bool, // True if the fenwick tree indexing is inverted due to it being a suffix tree
    feature: usize,
    current_feature_value: i32,
    next_feature_value: i32,
    dataview: &dataview::DataView<OT>,
    total_left: &mut OT::CostSumType, // Can be any costsum, for preventing allocations
    total_right: &mut OT::CostSumType, // Can be any costsum, for preventing allocations
) {
    for i in range {
        total_left.clear();
        let fenwick_idx = if reverse_fenwick {
            totals_left.len() - 1 - i
        } else {
            i
        };
        fenwick_prefix_sum(totals_left, fenwick_idx, total_left);

        total_right.clone_from(&totals[i]);
        *total_right -= &*total_left;
        best[i].add_candidate(
            branching_cost,
            total_left,
            total_right,
            feature,
            current_feature_value,
            next_feature_value,
            &dataview.dataset.internal_to_original_feature_value[feature],
        );
    }
}

/// TODO: this is a work in progress. Multiple things are broken.
/// Non-optimal results and crashes are expected.
/// For example, empty splits should not be allowed, but they are.
pub fn solve_d2<OT: OptimizationTask, SS: SearchStrategy>(
    dataview: &dataview::DataView<OT>,
    context: &SolveContext<OT, SS>,
) -> Arc<Tree<OT>> {
    let n_features = dataview.num_features();
    let branching_cost = context.task.branching_cost();

    // Reuse memory
    let mut feature_value_to_possible_split_idx: HashMap<i32, usize> = HashMap::with_capacity(0);
    let mut totals_left_i = Vec::with_capacity(0);
    let mut temp_total = dataview.cost_summer.clone();
    let mut temp_total2 = temp_total.clone();

    let mut best = Arc::new(Tree::Leaf(LeafNode {
        cost: dataview.cost_summer.cost(),
        label: dataview.cost_summer.label(),
    }));

    for f1 in 0..n_features {
        let n_splits = dataview.possible_split_values[f1].len();
        if n_splits == 0 {
            // No splits available for this feature, skip it.
            continue;
        }

        // Reset and reserve reused structs
        feature_value_to_possible_split_idx.clear();
        feature_value_to_possible_split_idx.reserve(n_splits);
        totals_left_i.clear();
        totals_left_i.reserve(n_splits);
        temp_total.clear();

        // Calculate the total cost on the left and right side when splitting at each possible split value.
        // Additionally create a lookup from feature_value to the last split that includes it. This may be > len
        let mut cur_threshold = Some(dataview.possible_split_values[f1][0].feature_value);
        let mut cur_i = 0;

        for &instance in dataview.instances_iter(f1) {
            if cur_threshold.is_some() && Some(instance.feature_value) > cur_threshold {
                // Note: this is not skipped at the end, because it is guaranteed that there will be one more instance
                // that is to the right of any possible split, otherwise the split is useless and would be excluded.
                totals_left_i.push(temp_total.clone());
                cur_i += 1;
                cur_threshold = dataview.possible_split_values[f1]
                    .get(cur_i)
                    .map(|x| x.feature_value);
            }

            // Update cumulative total
            temp_total += &dataview.dataset.instances[instance.instance_id];
            // If corresponding idx wasn't known yet, set it.
            feature_value_to_possible_split_idx.insert(instance.feature_value, cur_i);
        }

        let mut totals_right_i = Vec::with_capacity(n_splits);
        for total_left in &totals_left_i {
            let mut total_right = dataview.cost_summer.clone();
            total_right -= total_left;
            totals_right_i.push(total_right);
        }

        // TODO also account for not branching at all, i.e. just returning a leaf node.

        let mut best_left = Vec::with_capacity(n_splits);
        let mut best_right = Vec::with_capacity(n_splits);

        for i in 0..n_splits {
            best_left.push(D1ScoreTracker {
                leaf_cost: totals_left_i[i].cost(),
                feature: None,
                threshold: None,
                left_leaf: LeafNode {
                    cost: totals_left_i[i].cost(),
                    label: totals_left_i[i].label(),
                },
                right_leaf: None,
            });
            best_right.push(D1ScoreTracker {
                leaf_cost: totals_right_i[i].cost(),
                feature: None,
                threshold: None,
                left_leaf: LeafNode {
                    cost: totals_right_i[i].cost(),
                    label: totals_right_i[i].label(),
                },
                right_leaf: None,
            });
        }

        for f2 in 0..dataview.num_features() {
            if dataview.possible_split_values[f2].is_empty() {
                // No splits available for this feature, skip it.
                continue;
            }

            // These array do not directly contain the totals at the ith, but are instead fenwick trees.
            // Because of this, totals_right_left_i is indexed reversed.
            temp_total.clear(); // Used to clone zeros
            let mut totals_left_left_i = vec![temp_total.clone(); n_splits];
            let mut totals_right_left_i = vec![temp_total.clone(); n_splits];

            // It is possible that the for the left and the right side and different i's the previous is different.
            // The splits are still optimal, but possibly have less spacing between the features.
            // TODO: we could maintain a structure prev_left_feature_value_i and prev_right_feature_value_i to fix this.
            let mut previous = None;
            for &fv in dataview.instances_iter(f2) {
                // Check if we can split here.
                if let Some((prev_feature_value_1, prev_feature_value_2)) = previous {
                    if prev_feature_value_2 != fv.feature_value {
                        let first_split_idx_that_includes_it =
                            feature_value_to_possible_split_idx[&prev_feature_value_1];

                        // Check if any updated value results in the best tree.
                        // We do this before adding the value of the current instance so that we can check if this is a valid splitting point.
                        // TODO, if f1 == f2, we can be smarter.
                        // TODO, can we avoid doing this in some cases? E.g. similarity bound.
                        // Minimum difference is last best - UB, however it might be that the best solution on the left is
                        // improved by one, and the right by ten, so the next left still needs to be checked for a min improvement of 10.
                        // Maybe we can keep information from the previous cost update: If best_left[10] had min improvement of 10
                        // then in this iteration it will have a min improvement of at least 9. E.g. shrink the range.
                        update_lowest_cost(
                            branching_cost,
                            &mut best_left,
                            first_split_idx_that_includes_it..n_splits,
                            &totals_left_i,
                            &totals_left_left_i,
                            false,
                            f2,
                            prev_feature_value_2, // This may not be the previous in the current split, see TODO above
                            fv.feature_value,
                            dataview,
                            &mut temp_total,
                            &mut temp_total2,
                        );
                        update_lowest_cost(
                            branching_cost,
                            &mut best_right,
                            0..first_split_idx_that_includes_it,
                            &totals_right_i,
                            &totals_right_left_i,
                            true,
                            f2,
                            prev_feature_value_2, // This may not be the previous in the current split, see TODO above
                            fv.feature_value,
                            dataview,
                            &mut temp_total,
                            &mut temp_total2,
                        );
                    }
                }

                let feature_value_1 = dataview.dataset.feature_values[f1][fv.instance_id];
                previous = Some((feature_value_1, fv.feature_value));

                let instance = &dataview.dataset.instances[fv.instance_id];
                let first_split_idx_that_includes_it =
                    feature_value_to_possible_split_idx[&feature_value_1];

                // Update all possible feature splits in the root with this instance.
                // Take for example the first instance in f2 order. Say f1 = 3 for this instance.
                // Now this instance will fall on the left-left side for all feature tests greater or equal than 3.
                // And it will fall on the right-left side for all feature tests smaller than 3.
                //
                // If n_splits = 5, then totals_left_left_i should be updated at the last two splits 3 and 4 (index 0 and 1).
                // While totals_right_left will be updated at 0, 1 and 2 (same indices).

                // If it is n_splits, it is never on the left side.
                if first_split_idx_that_includes_it < n_splits {
                    // Add this instance value to all left-left totals of possible splits f1 <= x where feature_value_1 <= x (x >= feature_value_1)
                    fenwick_update(
                        &mut totals_left_left_i,
                        first_split_idx_that_includes_it,
                        instance,
                    );
                }

                // If it is 0, it is never on the right side.
                if first_split_idx_that_includes_it > 0 {
                    // Add this instance value to all right-left totals of possible splits f1 <= x where feature_value_1 > x (x < feature_value_1)
                    // We index reversed, because we want the suffix sum. Fenwick trees store the prefix sum.
                    let reversed_idx = (n_splits - 1) - (first_split_idx_that_includes_it - 1);
                    fenwick_update(&mut totals_right_left_i, reversed_idx, instance);
                }
            }
        }

        for (i, (left, right)) in best_left
            .into_iter()
            .zip(best_right.into_iter())
            .enumerate()
        {
            let ith_cost =
                left.total_cost(branching_cost) + right.total_cost(branching_cost) + branching_cost;
            if ith_cost.strictly_less_than(&best.cost()) {
                best = Arc::new(Tree::Branch(BranchNode {
                    cost: ith_cost,
                    split_feature: f1,
                    split_threshold: dataview.threshold_from_split(f1, i),
                    left_child: left.get_tree(branching_cost),
                    right_child: right.get_tree(branching_cost),
                }));
            }
        }
    }

    best
}

/// Exhaustive search for a node with depth two, given a fixed feature split.
pub fn solve_left_right<'a, OT: OptimizationTask, SS: SearchStrategy>(
    dataview: &dataview::DataView<OT>,
    context: &SolveContext<OT, SS>,
    feature: usize,
    split_index: usize,
) -> ExpandedQueueItem<'a, OT, SS> {
    let branching_cost = context.task.branching_cost();

    let split_value = dataview.possible_split_values[feature][split_index].feature_value;
    let mut total_right = dataview.cost_summer.clone();

    for instance in dataview.instances_iter(feature) {
        if dataview.dataset.feature_values[feature][instance.instance_id] > split_value {
            break;
        } else {
            total_right -= &dataview.dataset.instances[instance.instance_id];
        }
    }

    let mut total_left = dataview.cost_summer.clone();
    total_left -= &total_right;

    let left_tracker = D1ScoreTracker {
        leaf_cost: total_left.cost(),
        feature: None,
        threshold: None,
        left_leaf: LeafNode {
            cost: total_left.cost(),
            label: total_left.label(),
        },
        right_leaf: None,
    };
    let right_tracker = D1ScoreTracker {
        leaf_cost: total_right.cost(),
        feature: None,
        threshold: None,
        left_leaf: LeafNode {
            cost: total_right.cost(),
            label: total_right.label(),
        },
        right_leaf: None,
    };

    let left_done = left_tracker.is_optimal(branching_cost);
    let right_done = right_tracker.is_optimal(branching_cost);

    struct X<OT: OptimizationTask> {
        left_tracker: D1ScoreTracker<OT>,
        right_tracker: D1ScoreTracker<OT>,
        left_done: bool,
        right_done: bool,
    }

    impl<OT: OptimizationTask> Clone for X<OT> {
        fn clone(&self) -> Self {
            Self {
                left_tracker: self.left_tracker.clone(),
                right_tracker: self.right_tracker.clone(),
                left_done: self.left_done,
                right_done: self.right_done,
            }
        }
    }

    let state = X::<OT> {
        left_tracker,
        right_tracker,
        left_done,
        right_done,
    };

    let tl = ThreadLocal::new();

    (0..dataview.num_features())
        .into_par_iter()
        .map(|feature_2| {
            let mut state = tl
                .get_or(|| Mutex::new(state.clone()))
                .lock()
                .expect("Threadlocal state accessed once");

            if state.left_done && state.right_done {
                return true;
            }

            // init totals of the left left node and right left node to zero.
            let mut total_left_left = total_left.clone();
            total_left_left -= &total_left;
            let mut total_right_left = total_left.clone();
            total_right_left -= &total_left;

            let mut prev_left_feature_value = None;
            let mut prev_right_feature_value = None;

            // Keep costsums out of loop, so that we can .clone_from in the loop and avoid any allocations.
            let mut total_left_right = total_left.clone();
            let mut total_right_right = total_left.clone();

            for instance in dataview.instances_iter(feature_2) {
                if left_done && right_done {
                    return true;
                }

                let feature1_value = dataview.dataset.feature_values[feature][instance.instance_id];
                let feature2_value = instance.feature_value;
                if feature1_value <= split_value {
                    // Check if this is a point we can split at
                    if let Some(prev_left_feature_value) = prev_left_feature_value {
                        if !left_done && prev_left_feature_value != feature2_value {
                            total_left_right.clone_from(&total_left);
                            total_left_right -= &total_left_left;

                            state.left_tracker.add_candidate(
                                branching_cost,
                                &total_left_left,
                                &total_left_right,
                                feature_2,
                                prev_left_feature_value,
                                feature2_value,
                                &dataview.dataset.internal_to_original_feature_value[feature_2],
                            );
                            state.left_done |= state.left_tracker.is_optimal(branching_cost);
                        }
                    }
                    total_left_left += &dataview.dataset.instances[instance.instance_id];
                    prev_left_feature_value = Some(feature2_value);
                } else {
                    // Check if this is a point we can split at
                    if let Some(prev_right_feature_value) = prev_right_feature_value {
                        if !right_done && prev_right_feature_value != feature2_value {
                            total_right_right.clone_from(&total_right);
                            total_right_right -= &total_right_left;

                            state.right_tracker.add_candidate(
                                branching_cost,
                                &total_right_left,
                                &total_right_right,
                                feature_2,
                                prev_right_feature_value,
                                feature2_value,
                                &dataview.dataset.internal_to_original_feature_value[feature_2],
                            );
                            state.right_done |= state.right_tracker.is_optimal(branching_cost);
                        }
                    }
                    total_right_left += &dataview.dataset.instances[instance.instance_id];
                    prev_right_feature_value = Some(feature2_value);
                }
            }

            left_done && right_done
        })
        .any(|x| x); // The .any() terminates other running tasks early if left and right are optimal in some other task.

    let (left_tracker, right_tracker) = tl.iter().fold(
        (state.left_tracker.clone(), state.right_tracker.clone()),
        |(left_tracker, right_tracker), state| {
            let state = state.lock().expect("Thread local access");

            let best_left = if left_tracker
                .total_cost(branching_cost)
                .less_or_not_much_greater_than(&state.left_tracker.total_cost(branching_cost))
            {
                left_tracker
            } else {
                state.left_tracker.clone()
            };

            let best_right = if right_tracker
                .total_cost(branching_cost)
                .less_or_not_much_greater_than(&state.right_tracker.total_cost(branching_cost))
            {
                right_tracker
            } else {
                state.right_tracker.clone()
            };

            (best_left, best_right)
        },
    );

    let cost = left_tracker.total_cost(branching_cost)
        + right_tracker.total_cost(branching_cost)
        + branching_cost;

    ExpandedQueueItem::Solution(Arc::new(Tree::Branch(BranchNode {
        cost,
        split_feature: feature,
        split_threshold: dataview.threshold_from_split(feature, split_index),
        left_child: left_tracker.get_tree(branching_cost),
        right_child: right_tracker.get_tree(branching_cost),
    })))
}

/// Exhaustive search for a node with depth one.
pub fn solve_d1<OT: OptimizationTask, SS: SearchStrategy>(
    dataview: &dataview::DataView<OT>,
    context: &SolveContext<OT, SS>,
) -> Arc<Tree<OT>> {
    let branching_cost = context.task.branching_cost();
    let mut tracker = D1ScoreTracker {
        leaf_cost: dataview.cost_summer.cost(),
        feature: None,
        threshold: None,
        left_leaf: LeafNode {
            cost: dataview.cost_summer.cost(),
            label: dataview.cost_summer.label(),
        },
        right_leaf: None,
    };

    if tracker.is_optimal(branching_cost) {
        return tracker.get_tree(branching_cost);
    }

    // Keep costsums out of loop, so that we can .clone_from in the loop and avoid any allocations.
    let mut total_left = dataview.cost_summer.clone();
    let mut total_right = dataview.cost_summer.clone();

    for feature in 0..dataview.num_features() {
        // init total of the left node to zero.
        total_left.clone_from(&dataview.cost_summer);
        total_left -= &dataview.cost_summer;

        let mut prev_feature_value = None;

        for instance in dataview.instances_iter(feature) {
            let feature_value = instance.feature_value;
            // Check if this is a point we can split at
            if let Some(prev_feature_value) = prev_feature_value {
                if prev_feature_value != feature_value {
                    total_right.clone_from(&dataview.cost_summer);
                    total_right -= &total_left;

                    tracker.add_candidate(
                        branching_cost,
                        &total_left,
                        &total_right,
                        feature,
                        prev_feature_value,
                        feature_value,
                        &dataview.dataset.internal_to_original_feature_value[feature],
                    );
                    if tracker.is_optimal(branching_cost) {
                        return tracker.get_tree(branching_cost);
                    }
                }
            }
            total_left += &dataview.dataset.instances[instance.instance_id];
            prev_feature_value = Some(feature_value);
        }
    }

    tracker.get_tree(branching_cost)
}

struct D1ScoreTracker<OT: OptimizationTask> {
    leaf_cost: OT::CostType,
    feature: Option<usize>,
    threshold: Option<f64>,
    /// The left leaf is not an option as it is used to store the label in case there is no split.
    left_leaf: LeafNode<OT>,
    right_leaf: Option<LeafNode<OT>>,
}

impl<OT: OptimizationTask> Clone for D1ScoreTracker<OT> {
    fn clone(&self) -> Self {
        Self {
            leaf_cost: self.leaf_cost,
            feature: self.feature,
            threshold: self.threshold,
            left_leaf: self.left_leaf.clone(),
            right_leaf: self.right_leaf.clone(),
        }
    }
}

impl<OT: OptimizationTask> D1ScoreTracker<OT> {
    fn is_optimal(&self, branching_cost: OT::CostType) -> bool {
        // If we do branch, then it is optimal if the cost after branching is zero. (Since we have already excluded not branching)
        // If we do not branch (yet), this is optimal if branching cannot improve the cost.
        self.leaf_cost.is_zero()
            || (self.feature.is_none()
                && self
                    .leaf_cost
                    .less_or_not_much_greater_than(&branching_cost))
    }

    fn total_cost(&self, branching_cost: OT::CostType) -> OT::CostType {
        if self.feature.is_some() {
            self.leaf_cost + branching_cost
        } else {
            self.leaf_cost
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn add_candidate(
        &mut self,
        branching_cost: OT::CostType,
        costsum_left: &OT::CostSumType,
        costsum_right: &OT::CostSumType,
        feature: usize,
        current_feature_value: i32,
        next_feature_value: i32,
        feature_value_to_threshold: &[f64],
    ) {
        let cost_left = costsum_left.cost();
        let cost_right = costsum_right.cost();
        let leaf_cost = cost_left + cost_right;
        let total_cost = leaf_cost + branching_cost;

        if total_cost.strictly_less_than(&self.total_cost(branching_cost)) {
            let current_threshold = feature_value_to_threshold[current_feature_value as usize];
            let next_threshold = feature_value_to_threshold[next_feature_value as usize];

            self.leaf_cost = leaf_cost;
            self.feature = Some(feature);
            self.threshold = Some((current_threshold + next_threshold) / 2.0);
            self.left_leaf = LeafNode {
                cost: cost_left,
                label: costsum_left.label(),
            };
            self.right_leaf = Some(LeafNode {
                cost: cost_right,
                label: costsum_right.label(),
            });
        }
    }

    fn get_tree(self, branching_cost: OT::CostType) -> Arc<Tree<OT>> {
        let tree = if self.feature.is_none() {
            Tree::Leaf(LeafNode {
                cost: self.total_cost(branching_cost),
                label: self.left_leaf.label,
            })
        } else {
            Tree::Branch(BranchNode {
                cost: self.total_cost(branching_cost),
                split_feature: self.feature.unwrap(),
                split_threshold: self.threshold.unwrap(),
                left_child: Arc::new(Tree::Leaf(self.left_leaf)),
                right_child: Arc::new(Tree::Leaf(self.right_leaf.unwrap())),
            })
        };

        Arc::new(tree)
    }
}
