//! Superbubble detection over the POA graph.
//!
//! A *superbubble* is a single-entry, single-exit subgraph of a DAG: a boundary pair `(entry,
//! exit)` such that the vertices between them form a self-contained bubble (every path leaving
//! `entry` reconverges at `exit`, with no edges escaping or entering the region except through the
//! two boundaries). Superbubbles capture the local alignment ambiguities (substitutions, small
//! indels) a POA graph encodes.
//!
//! [`Graph::superbubbles`] finds them in near-linear time (`O(E + V log V)`) with the Brankovic et
//! al. (2016) superbubble algorithm over the DAG augmented with a virtual super-source and
//! super-sink (see [`SuperbubbleEnd`]), so bubbles bounded by the graph's head or tail are still
//! reported. The algorithm is a pure read-only traversal of the adjacency [`Graph`] already exposes.

use crate::graph::{Graph, NodeId};

/// A boundary of a superbubble reported by [`Graph::superbubbles`].
///
/// Superbubble detection augments the POA DAG with a virtual super-source (preceding every real
/// source node — a node with no in-edges) and a virtual super-sink (following every real sink node
/// — a node with no out-edges), so a superbubble bounded by the graph's head or tail is reported
/// with a virtual boundary rather than being missed. A caller that works in MSA-column space
/// (see [`Graph::msa_columns`]) maps [`SuperbubbleEnd::Source`] to column `0`,
/// [`SuperbubbleEnd::Sink`] to column `n_columns`, and [`SuperbubbleEnd::Node`]`(id)` to that
/// node's column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuperbubbleEnd {
    /// The virtual super-source: precedes every real source node (a node with no in-edges).
    Source,
    /// A real graph node.
    Node(NodeId),
    /// The virtual super-sink: follows every real sink node (a node with no out-edges).
    Sink,
}

impl SuperbubbleEnd {
    /// Maps an augmented id (as used by [`Graph::superbubbles`], with `source == n` and
    /// `sink == n + 1`, where `n` is the real-node count) back to a [`SuperbubbleEnd`].
    fn from_augmented(id: usize, n: usize) -> SuperbubbleEnd {
        if id == n {
            SuperbubbleEnd::Source
        } else if id == n + 1 {
            SuperbubbleEnd::Sink
        } else {
            SuperbubbleEnd::Node(NodeId(id as u32))
        }
    }
}

/// One detected superbubble as its `(entry, exit)` boundary pair, produced by
/// [`Graph::superbubbles`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Superbubble {
    /// The entrance of the superbubble.
    pub entry: SuperbubbleEnd,
    /// The exit of the superbubble.
    pub exit: SuperbubbleEnd,
}

impl Graph {
    /// All superbubbles in the POA DAG — one *smallest* superbubble per entrance that starts one —
    /// found by the Brankovic–Iliopoulos–Kundu–Mohamed–Pissis–Vayani (2016) superbubble algorithm
    /// over the DAG augmented with a virtual super-source and virtual super-sink (see
    /// [`SuperbubbleEnd`]).
    ///
    /// Entrances are reported in topological order: the virtual super-source first (so head
    /// superbubbles spanning several real source nodes are found), then every real node in
    /// [`Graph::rank_order`] order. An entrance that starts no superbubble contributes nothing.
    ///
    /// Results are **raw**: not deduplicated, not nested-filtered, not empty-interior-filtered — the
    /// caller applies whatever domain filtering it needs. In particular the degenerate
    /// adjacent-boundary pairs (a linear stretch's `(Source, first)`, each `(node_i, node_{i+1})`,
    /// and `(last, Sink)`) ARE included. Parallel edges between a node pair count as a single
    /// adjacency. The graph is assumed acyclic (a POA DAG); on a cyclic graph (which cannot arise
    /// from `add_alignment`) no superbubbles are reported.
    ///
    /// The augmentation gives real nodes the ids `0..n`, the virtual super-source id `n`, and the
    /// virtual super-sink id `n + 1`, where `n == self.num_nodes()`. Each returned boundary maps
    /// that id back to a [`SuperbubbleEnd`].
    ///
    /// # Complexity
    /// `O(E + V log V)`: a single DFS topological pass and a candidate sweep whose total
    /// entrance-scanning work is amortized linear, plus `O(V log V)` sparse-table range-min/max
    /// preprocessing (the one super-linear term — the paper's `O(V + E)` bound needs a
    /// linear-preprocessing constant-time RMQ, not built here since the `log V` factor is
    /// negligible for POA-sized graphs). Still far below the previous per-entrance frontier walk's
    /// `O(V · (V + E))`.
    #[must_use]
    pub fn superbubbles(&self) -> Vec<Superbubble> {
        let n = self.num_nodes();
        let source = n;
        let (successors, predecessors) = self.augmented_adjacency();

        // `exit_of[entrance]` is the exit of the unique superbubble that augmented-id `entrance`
        // starts (if any). Emitting in the documented `[source, rank_order]` order keeps the
        // returned `Vec` identical (order and content) to the per-entrance formulation.
        let exit_of = brankovic_superbubbles(&successors, &predecessors);
        let mut superbubbles: Vec<Superbubble> = Vec::new();
        for entrance in
            std::iter::once(source).chain(self.rank_order().iter().map(|id| id.0 as usize))
        {
            if let Some(exit) = exit_of[entrance] {
                superbubbles.push(Superbubble {
                    entry: SuperbubbleEnd::from_augmented(entrance, n),
                    exit: SuperbubbleEnd::from_augmented(exit, n),
                });
            }
        }
        superbubbles
    }

    /// Builds the deduplicated augmented adjacency used by [`Graph::superbubbles`], as
    /// `(successors, predecessors)` indexed by augmented id over `0..num_nodes() + 2`.
    ///
    /// Real nodes keep their ids `0..n`; `source = n` gains an edge to every in-edge-free real
    /// node, and `sink = n + 1` gains an edge from every out-edge-free real node. Parallel edges
    /// between a real node pair collapse to a single adjacency (via [`Node::successors`] /
    /// [`Node::predecessors`]). `successors[sink]` and `predecessors[source]` stay empty.
    fn augmented_adjacency(&self) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let n = self.num_nodes();
        let source = n;
        let sink = n + 1;
        let total = n + 2;

        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); total];
        let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); total];
        for i in 0..n {
            let node = &self.nodes()[i];
            successors[i] = node
                .successors(self)
                .iter()
                .map(|id| id.0 as usize)
                .collect();
            predecessors[i] = node
                .predecessors(self)
                .iter()
                .map(|id| id.0 as usize)
                .collect();
        }
        // Wire the virtual super-source into every real source node and the virtual super-sink out
        // of every real sink node. Each real node is visited once, so no dedup is needed here.
        for i in 0..n {
            let node = &self.nodes()[i];
            if node.inedges.is_empty() {
                successors[source].push(i);
                predecessors[i].push(source);
            }
            if node.outedges.is_empty() {
                successors[i].push(sink);
                predecessors[sink].push(i);
            }
        }
        (successors, predecessors)
    }
}

// ---- Brankovic near-linear superbubble detector -------------------------------------------------
//
// A faithful reimplementation of the DAG superbubble algorithm of Brankovic, Iliopoulos, Kundu,
// Mohamed, Pissis & Vayani (Theoretical Computer Science, 2016). The paper attains O(V + E) with a
// linear-preprocessing constant-time RMQ; this port uses a sparse-table RMQ (O(V log V) build,
// O(1) query), so it is O(E + V log V) — negligibly above linear for POA graphs. It runs over the
// augmented
// adjacency from `Graph::augmented_adjacency` and returns, per augmented entrance id, the exit of
// the superbubble it starts. It replaces the earlier per-entrance frontier walk with no change to
// the reported set (validated against that walk and an independent brute-force oracle over
// hundreds of thousands of random DAGs and a large real POA corpus).

/// Reduce over the exit of the superbubble each augmented vertex starts (`None` if it starts none).
///
/// `successors`/`predecessors` are the deduplicated augmented adjacency. On a cyclic graph (which a
/// POA DAG never is) every entry is `None`.
fn brankovic_superbubbles(
    successors: &[Vec<usize>],
    predecessors: &[Vec<usize>],
) -> Vec<Option<usize>> {
    let total = successors.len();
    let Some((vertex, ord)) = reverse_post_order(successors, predecessors) else {
        return vec![None; total]; // cyclic input: unspecified, report nothing.
    };

    // `out_parent[pos]` = min topological position among the parents of `vertex[pos]` (sentinel
    // `total` when it has none — only the virtual source, which the validation never queries as an
    // interior parent). `out_child[pos]` = max position among children (sentinel `0` for the sink).
    let mut out_parent = vec![0usize; total];
    let mut out_child = vec![0usize; total];
    for (pos, &v) in vertex.iter().enumerate() {
        out_parent[pos] = predecessors[v]
            .iter()
            .map(|&u| ord[u])
            .min()
            .unwrap_or(total);
        out_child[pos] = successors[v].iter().map(|&w| ord[w]).max().unwrap_or(0);
    }

    let mut detector = SuperbubbleDetector::new(&vertex, &ord, &out_parent, &out_child);
    detector.run(successors, predecessors);
    detector.exit_of
}

/// DFS reverse-post-order topological order of the augmented adjacency, as `(vertex_at_pos,
/// pos_of_vertex)`. Returns `None` on a cycle.
///
/// Brankovic's range-min/range-max interval test requires a *DFS* order (not Kahn/BFS): it relies
/// on each superbubble occupying a contiguous topological interval, which reverse-post-order gives
/// but an arbitrary topological order does not.
fn reverse_post_order(
    successors: &[Vec<usize>],
    predecessors: &[Vec<usize>],
) -> Option<(Vec<usize>, Vec<usize>)> {
    let total = successors.len();
    let mut color = vec![0u8; total]; // 0 = unseen, 1 = on-stack, 2 = done
    let mut post: Vec<usize> = Vec::with_capacity(total);
    // Root the DFS at every in-edge-free vertex (the augmented source covers the rest); fall back
    // to all vertices so any component is reached even on malformed input.
    let roots = (0..total).filter(|&v| predecessors[v].is_empty());
    for start in roots.chain(0..total) {
        if color[start] != 0 {
            continue;
        }
        color[start] = 1;
        let mut stack: Vec<(usize, usize)> = vec![(start, 0)]; // (vertex, next child index)
        while let Some(&(v, child_idx)) = stack.last() {
            if child_idx < successors[v].len() {
                stack.last_mut().unwrap().1 += 1;
                let w = successors[v][child_idx];
                match color[w] {
                    0 => {
                        color[w] = 1;
                        stack.push((w, 0));
                    }
                    1 => return None, // on-stack ⇒ back edge ⇒ cycle
                    _ => {}
                }
            } else {
                color[v] = 2;
                post.push(v);
                stack.pop();
            }
        }
    }
    if post.len() != total {
        return None;
    }
    post.reverse(); // reverse post-order is a topological order
    let mut ord = vec![0usize; total];
    for (pos, &v) in post.iter().enumerate() {
        ord[v] = pos;
    }
    Some((post, ord))
}

/// Range-min / range-max sparse table over a slice of positions: `O(n log n)` build, `O(1)` query.
struct Rmq {
    table: Vec<Vec<usize>>,
    is_min: bool,
}

impl Rmq {
    fn new(values: &[usize], is_min: bool) -> Rmq {
        let n = values.len();
        let levels = usize::BITS as usize - n.max(1).leading_zeros() as usize;
        let mut table = vec![values.to_vec()];
        for level in 1..levels {
            let prev = &table[level - 1];
            let half = 1usize << (level - 1);
            let span = 1usize << level;
            let row: Vec<usize> = (0..n)
                .map(|i| {
                    if i + span <= n {
                        let (x, y) = (prev[i], prev[i + half]);
                        if is_min {
                            x.min(y)
                        } else {
                            x.max(y)
                        }
                    } else {
                        prev[i]
                    }
                })
                .collect();
            table.push(row);
        }
        Rmq { table, is_min }
    }

    /// Reduce over the inclusive range `[l, r]`; the caller guarantees `l <= r < n`.
    fn query(&self, l: usize, r: usize) -> usize {
        let len = r - l + 1;
        let level = usize::BITS as usize - 1 - len.leading_zeros() as usize;
        let (x, y) = (
            self.table[level][l],
            self.table[level][r + 1 - (1 << level)],
        );
        if self.is_min {
            x.min(y)
        } else {
            x.max(y)
        }
    }
}

/// One candidate in the Brankovic sweep: a vertex that is a possible entrance or exit. A vertex
/// that is both is represented by two candidates (exit first), matching the reference.
struct Candidate {
    vertex: usize,
    is_entrance: bool,
    /// The nearest entrance candidate index at or before this candidate (`-1` if none) — read only
    /// for exit candidates, as their `pvsEntrance`.
    pvs_entrance: i64,
}

/// State for the Brankovic candidate-pairing sweep. The candidate list is append-in-topological-
/// order with tail-only deletions, i.e. a stack, modelled here as a `Vec` plus a live-count `tail`.
struct SuperbubbleDetector<'a> {
    ord: &'a [usize],
    vertex: &'a [usize],
    rmq_out_parent: Rmq, // range-min
    rmq_out_child: Rmq,  // range-max
    candidates: Vec<Candidate>,
    /// Per vertex: the nearest entrance candidate index at or before it (`-1` if none).
    pvs_entrance_of_vertex: Vec<i64>,
    /// Per vertex: the exit vertex last transitioned to from it, to break alternative-entrance
    /// cycles (`-1` unmarked).
    mark: Vec<i64>,
    /// Per augmented vertex: the exit of the superbubble it starts, if any.
    exit_of: Vec<Option<usize>>,
    /// Number of live candidates (`candidates[..tail]`); deletions only ever pop the tail.
    tail: usize,
}

impl<'a> SuperbubbleDetector<'a> {
    fn new(
        vertex: &'a [usize],
        ord: &'a [usize],
        out_parent: &[usize],
        out_child: &[usize],
    ) -> Self {
        let total = vertex.len();
        // The candidate list and `pvs_entrance_of_vertex` are built in `run`, which has the
        // adjacency needed for the degree predicates.
        SuperbubbleDetector {
            ord,
            vertex,
            rmq_out_parent: Rmq::new(out_parent, true),
            rmq_out_child: Rmq::new(out_child, false),
            candidates: Vec::new(),
            pvs_entrance_of_vertex: vec![-1i64; total],
            mark: vec![-1i64; total],
            exit_of: vec![None; total],
            tail: 0,
        }
    }

    fn run(&mut self, successors: &[Vec<usize>], predecessors: &[Vec<usize>]) {
        // Build the candidate list in topological order (exit entry before entrance entry for a
        // both-candidate vertex), tracking each vertex's previous entrance candidate.
        let mut pvs_entrance: i64 = -1;
        for &v in self.vertex {
            if predecessors[v].iter().any(|&u| successors[u].len() == 1) {
                self.candidates.push(Candidate {
                    vertex: v,
                    is_entrance: false,
                    pvs_entrance,
                });
            }
            if successors[v].iter().any(|&w| predecessors[w].len() == 1) {
                self.candidates.push(Candidate {
                    vertex: v,
                    is_entrance: true,
                    pvs_entrance: -1,
                });
                pvs_entrance = (self.candidates.len() - 1) as i64;
            }
            self.pvs_entrance_of_vertex[v] = pvs_entrance;
        }

        self.tail = self.candidates.len();
        while self.tail > 0 {
            let t = self.tail - 1;
            if self.candidates[t].is_entrance {
                self.tail -= 1;
            } else {
                self.report(0, t);
            }
        }
    }

    /// Returns the candidate index `s` (a valid entrance for `end`), `-1` for "no superbubble here"
    /// (`null`), or an alternative inner entrance candidate index to try next.
    fn validate(&self, s_idx: usize, end_idx: usize) -> i64 {
        let start = self.ord[self.candidates[s_idx].vertex];
        let end = self.ord[self.candidates[end_idx].vertex];
        if self.rmq_out_child.query(start, end - 1) != end {
            return -1; // an interior child escapes past the exit ⇒ not a superbubble
        }
        let out_parent = self.rmq_out_parent.query(start + 1, end);
        if out_parent == start {
            return s_idx as i64; // start is the sole entrance ⇒ valid
        }
        self.pvs_entrance_of_vertex[self.vertex[out_parent]] // inner alternative entrance (or -1)
    }

    /// Reports the superbubble entered at (an entrance at or after) `start_idx` and exited at
    /// `exit_idx`, then recurses into its interior for nested superbubbles.
    ///
    /// Recursion depth equals the superbubble *nesting* depth, which is shallow for POA graphs
    /// (bounded by distinct-column structure); an adversarial deeply-nested graph would need an
    /// explicit work stack instead.
    fn report(&mut self, start_idx: usize, exit_idx: usize) {
        if self.ord[self.candidates[start_idx].vertex] >= self.ord[self.candidates[exit_idx].vertex]
        {
            self.tail -= 1;
            return;
        }
        // Walk `s` back through alternative (inner) entrances; report ONLY when `validate` actually
        // returns `s` (⟨s, exit⟩ is valid). Exiting via the range condition must not report.
        let mut s: i64 = self.candidates[exit_idx].pvs_entrance;
        let mut found = false;
        while s != -1
            && self.ord[self.candidates[s as usize].vertex]
                >= self.ord[self.candidates[start_idx].vertex]
        {
            let valid = self.validate(s as usize, exit_idx);
            if valid == s {
                found = true;
                break;
            }
            let s_vertex = self.candidates[s as usize].vertex;
            if valid == -1 || self.candidates[valid as usize].vertex as i64 == self.mark[s_vertex] {
                break;
            }
            self.mark[s_vertex] = self.candidates[valid as usize].vertex as i64;
            s = valid;
        }

        let exit_vertex = self.candidates[exit_idx].vertex;
        self.tail -= 1; // delete the exit we processed
        if found {
            self.exit_of[self.candidates[s as usize].vertex] = Some(exit_vertex);
            // Recurse into the interior for nested superbubbles, popping entrance candidates.
            while self.tail > 0 {
                let next_idx = self.tail - 1;
                if next_idx == s as usize {
                    break;
                }
                if self.candidates[next_idx].is_entrance {
                    self.tail -= 1;
                } else {
                    self.report(s as usize + 1, next_idx);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Edge, EdgeId};
    use proptest::prelude::*;
    use std::collections::HashSet;

    /// The Onodera–Sadakane per-entrance frontier walk that `superbubbles()` used before the
    /// Brankovic swap, kept as an independent `#[cfg(test)]` differential oracle. Returns the
    /// smallest superbubble `entrance` starts, or `None`. Terminates on any finite graph: a node is
    /// pushed only once all its predecessors are visited, so cyclic nodes are never settled.
    fn superbubble_from(
        successors: &[Vec<usize>],
        predecessors: &[Vec<usize>],
        entrance: usize,
    ) -> Option<(usize, usize)> {
        let mut seen: HashSet<usize> = HashSet::new();
        let mut visited: HashSet<usize> = HashSet::new();
        let mut stack: Vec<usize> = vec![entrance];
        while let Some(v) = stack.pop() {
            visited.insert(v);
            seen.remove(&v);
            let children = &successors[v];
            if children.is_empty() {
                return None;
            }
            for &child in children {
                if child == entrance {
                    return None;
                }
                seen.insert(child);
                if predecessors[child].iter().all(|p| visited.contains(p)) {
                    stack.push(child);
                }
            }
            if stack.len() == 1 && seen.len() == 1 && seen.contains(&stack[0]) {
                return Some((entrance, stack[0]));
            }
        }
        None
    }

    /// The full `superbubbles()` output computed via the frontier-walk oracle instead of Brankovic:
    /// the walk run over `[source, rank_order]` entrances, mapped to `Superbubble`s. Used to assert
    /// the Brankovic engine reproduces the walk's output exactly (order and content).
    fn walk_superbubbles(graph: &Graph) -> Vec<Superbubble> {
        let n = graph.num_nodes();
        let source = n;
        let (successors, predecessors) = graph.augmented_adjacency();
        std::iter::once(source)
            .chain(graph.rank_order().iter().map(|id| id.0 as usize))
            .filter_map(|entrance| superbubble_from(&successors, &predecessors, entrance))
            .map(|(entry, exit)| Superbubble {
                entry: SuperbubbleEnd::from_augmented(entry, n),
                exit: SuperbubbleEnd::from_augmented(exit, n),
            })
            .collect()
    }

    // These build small DAGs directly in the arena (`add_node` + `add_edge` + `topological_sort`)
    // for exact control over node ids and topology, then assert the RAW `superbubbles()` output,
    // including the trivial adjacent-boundary pairs the Onodera-Sadakane walk emits (see the raw-
    // output contract on `Graph::superbubbles`). Trivial pairs are the graph's head `(Source,
    // first)`, tail `(last, Sink)`, and each unbranched `(node_i, node_{i+1})` step; only the
    // consumer's column-space filter distinguishes "meaningful" bubbles, so we replicate that
    // filter test-locally in `meaningful_bubbles` to show intent.

    /// Adds `edges` (as `(tail, head)` node-id pairs) with unit weight/label, then topologically
    /// sorts so `rank_order()` (the entrance order) is populated.
    fn wire(graph: &mut Graph, edges: &[(u32, u32)]) {
        for &(tail, head) in edges {
            graph.add_edge(NodeId(tail), NodeId(head), 1, 0);
        }
        graph.topological_sort();
    }

    /// Builds a graph of `n_nodes` codeless nodes wired by `edges`.
    fn build(n_nodes: u32, edges: &[(u32, u32)]) -> Graph {
        let mut graph = Graph::new();
        for _ in 0..n_nodes {
            graph.add_node(0);
        }
        wire(&mut graph, edges);
        graph
    }

    /// Short constructors for the expected `SuperbubbleEnd`s.
    fn src() -> SuperbubbleEnd {
        SuperbubbleEnd::Source
    }
    fn snk() -> SuperbubbleEnd {
        SuperbubbleEnd::Sink
    }
    fn nd(id: u32) -> SuperbubbleEnd {
        SuperbubbleEnd::Node(NodeId(id))
    }
    fn sb(entry: SuperbubbleEnd, exit: SuperbubbleEnd) -> Superbubble {
        Superbubble { entry, exit }
    }

    /// The consumer's column-space "real bubble" filter (relay §6), applied test-locally: map each
    /// boundary to an MSA column (`Source` -> 0, `Sink` -> `n_columns`, `Node` -> its column) and
    /// keep only pairs with a non-empty interior (`exit_col > entry_col + 1`).
    fn meaningful_bubbles(graph: &Graph) -> Vec<(usize, usize)> {
        let (node_to_col, n_columns) = graph.msa_columns();
        let col_of = |end: SuperbubbleEnd| -> usize {
            match end {
                SuperbubbleEnd::Source => 0,
                SuperbubbleEnd::Sink => n_columns as usize,
                SuperbubbleEnd::Node(id) => node_to_col[id.0 as usize] as usize,
            }
        };
        graph
            .superbubbles()
            .into_iter()
            .map(|bubble| (col_of(bubble.entry), col_of(bubble.exit)))
            .filter(|(entry_col, exit_col)| *exit_col > *entry_col + 1)
            .collect()
    }

    /// A linear chain yields only trivial adjacent pairs — one per augmented edge — and no
    /// meaningful superbubble. Pins that trivials ARE emitted (raw-output contract).
    #[test]
    fn superbubbles_linear_chain_yields_only_trivial_pairs() {
        let graph = build(4, &[(0, 1), (1, 2), (2, 3)]);
        assert_eq!(
            graph.superbubbles(),
            vec![
                sb(src(), nd(0)),
                sb(nd(0), nd(1)),
                sb(nd(1), nd(2)),
                sb(nd(2), nd(3)),
                sb(nd(3), snk()),
            ],
        );
        assert!(
            meaningful_bubbles(&graph).is_empty(),
            "a linear chain has no non-empty-interior bubble"
        );
    }

    /// A single diamond `a->{b,c}->d` reports the meaningful `Node(a)..Node(d)` bubble flanked by
    /// the trivial `Source`/`Sink` boundary pairs.
    #[test]
    fn superbubbles_single_diamond_reports_the_bubble_plus_trivial_boundaries() {
        // n0=a, n1=b, n2=c, n3=d
        let graph = build(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        assert_eq!(
            graph.superbubbles(),
            vec![sb(src(), nd(0)), sb(nd(0), nd(3)), sb(nd(3), snk())],
        );
        // Exactly one survives the interior-non-empty filter: the diamond itself.
        assert_eq!(meaningful_bubbles(&graph).len(), 1);
    }

    /// Regression: two source nodes converging at a real node produce a superbubble whose `entry`
    /// is the virtual `Source`. Without the super-source augmentation this bubble is missed
    /// entirely (a multi-node head has no single real reconvergence entrance).
    #[test]
    fn superbubbles_multi_source_head_reports_source_entry() {
        // s0=0, s1=1 both -> m=2 -> e=3
        let graph = build(4, &[(0, 2), (1, 2), (2, 3)]);
        let bubbles = graph.superbubbles();
        assert!(
            bubbles.contains(&sb(src(), nd(2))),
            "expected a Source-entry bubble exiting at the reconvergence node m; got {bubbles:?}"
        );
        // And it is meaningful (its interior holds the two source nodes).
        let (node_to_col, _) = graph.msa_columns();
        let m_col = node_to_col[2] as usize;
        assert!(
            m_col > 1,
            "Source(0)..m must have a non-empty column interior"
        );
    }

    /// A branch whose two arms are both real sinks produces a superbubble whose `exit` is the
    /// virtual `Sink`, with a non-empty interior (the two sink arms).
    #[test]
    fn superbubbles_multi_sink_tail_reports_sink_exit() {
        // h=0 -> b=1 -> {t0=2, t1=3}, both t0/t1 are sinks
        let graph = build(4, &[(0, 1), (1, 2), (1, 3)]);
        let bubbles = graph.superbubbles();
        assert!(
            bubbles.contains(&sb(nd(1), snk())),
            "expected a Sink-exit bubble entering at the branch node b; got {bubbles:?}"
        );
        assert_eq!(
            meaningful_bubbles(&graph),
            {
                let (node_to_col, n_columns) = graph.msa_columns();
                vec![(node_to_col[1] as usize, n_columns as usize)]
            },
            "the branch-to-both-sinks bubble is the only meaningful one"
        );
    }

    /// Nested bubbles: a diamond whose top arm contains an inner diamond. Both the inner and the
    /// outer smallest-per-entrance superbubbles are returned (raw output is not nested-filtered).
    #[test]
    fn superbubbles_nested_returns_both_inner_and_outer() {
        // Outer diamond 0..6; inner diamond 1..4 sits on the top arm; bottom arm is 0->5->6.
        //   0 -> 1, 0 -> 5
        //   1 -> 2, 1 -> 3, 2 -> 4, 3 -> 4   (inner diamond, entry 1, exit 4)
        //   4 -> 6, 5 -> 6                   (arms reconverge at outer exit 6)
        let graph = build(
            7,
            &[
                (0, 1),
                (0, 5),
                (1, 2),
                (1, 3),
                (2, 4),
                (3, 4),
                (4, 6),
                (5, 6),
            ],
        );
        let bubbles = graph.superbubbles();
        assert!(
            bubbles.contains(&sb(nd(1), nd(4))),
            "inner diamond (1..4) must appear; got {bubbles:?}"
        );
        assert!(
            bubbles.contains(&sb(nd(0), nd(6))),
            "outer diamond (0..6) must appear; got {bubbles:?}"
        );
        // Pin the FULL raw output (not just the two contains checks) against the independent
        // oracle. This 7-node graph is outside the property test's `n <= 6` range, so without this
        // equality a spurious or wrong trivial bubble here would go uncaught.
        assert_eq!(bubbles, brute_force_superbubbles(&graph));
    }

    /// Parallel edges between the same node pair dedup to one adjacency: the graph behaves like a
    /// simple 2-node chain (no spurious bubble beyond the trivial pairs). Built by pushing two
    /// edges into the arena directly, since `add_edge` folds a repeated `(tail, head)`.
    #[test]
    fn superbubbles_parallel_edges_dedup_to_one_adjacency() {
        let mut graph = Graph::new();
        graph.add_node(0); // n0
        graph.add_node(0); // n1
                           // `add_edge` folds a repeated (tail, head), so push two genuine parallel edges into the
                           // arena directly to exercise the adjacency dedup.
        for label in 0..2u32 {
            let edge_id = EdgeId(graph.edges().len() as u32);
            graph.edges.push(Edge {
                tail: NodeId(0),
                head: NodeId(1),
                labels: vec![label],
                weight: 1,
            });
            graph.nodes[0].outedges.push(edge_id);
            graph.nodes[1].inedges.push(edge_id);
        }
        graph.topological_sort();

        // Two genuine parallel edges exist, but they collapse to a single successor/predecessor.
        assert_eq!(graph.node(NodeId(0)).outedges.len(), 2);
        assert_eq!(graph.node(NodeId(0)).successors(&graph), vec![NodeId(1)]);
        assert_eq!(graph.node(NodeId(1)).predecessors(&graph), vec![NodeId(0)]);
        // No spurious bubble: identical to a plain 2-node chain.
        assert_eq!(
            graph.superbubbles(),
            vec![sb(src(), nd(0)), sb(nd(0), nd(1)), sb(nd(1), snk())],
        );
    }

    /// Whole-graph augmentation: a graph that is entirely one head-to-tail bubble (two disjoint
    /// source->sink paths reconverging only at the virtual boundaries) reports a `Source..Sink`
    /// superbubble with a non-empty interior.
    #[test]
    fn superbubbles_whole_graph_bubble_reports_source_to_sink() {
        // Path A: 0 -> 1 -> 2 ; Path B: 3 -> 4 -> 5 ; disjoint, no shared real node.
        let graph = build(6, &[(0, 1), (1, 2), (3, 4), (4, 5)]);
        let bubbles = graph.superbubbles();
        assert!(
            bubbles.contains(&sb(src(), snk())),
            "expected a whole-graph Source..Sink bubble; got {bubbles:?}"
        );
        assert!(
            meaningful_bubbles(&graph).contains(&(0, graph.msa_columns().1 as usize)),
            "the Source..Sink bubble spans the whole column range and is meaningful"
        );
    }

    /// Determinism: the same graph yields the identical `Vec<Superbubble>` on repeated calls.
    #[test]
    fn superbubbles_is_deterministic() {
        let graph = build(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        assert_eq!(graph.superbubbles(), graph.superbubbles());
    }

    /// An empty graph has no superbubbles (the walk from the lone virtual source dead-ends).
    #[test]
    fn superbubbles_empty_graph_is_empty() {
        assert!(Graph::new().superbubbles().is_empty());
    }

    /// `Node::successors` / `predecessors` dedup to the first occurrence of each neighbor and
    /// preserve edge-insertion order. Uses a non-palindromic neighbor order with an interleaved
    /// duplicate, so the full-vec assertion fails for both a reversed-iteration and a
    /// last-occurrence-wins implementation. (The superbubble walk hashes neighbors into sets, so it
    /// cannot observe this order — only a direct assertion pins the documented contract.)
    #[test]
    fn successors_and_predecessors_preserve_first_occurrence_order() {
        let mut graph = Graph::new();
        for _ in 0..5 {
            graph.add_node(0); // nodes 0..=4
        }
        // Push edges straight into the arena so `add_edge`'s fold can't reorder/collapse them
        // before we observe dedup order.
        let mut push = |tail: u32, head: u32| {
            let id = EdgeId(graph.edges().len() as u32);
            graph.edges.push(Edge {
                tail: NodeId(tail),
                head: NodeId(head),
                labels: vec![0],
                weight: 1,
            });
            graph.nodes[tail as usize].outedges.push(id);
            graph.nodes[head as usize].inedges.push(id);
        };
        // node 0 out-edges to heads [2, 1, 3, 1]; node 4 in-edges from tails [3, 1, 2, 1]. The two
        // fixtures share no endpoint that would cross-pollute the asserted adjacency of node 0 / 4.
        for head in [2u32, 1, 3, 1] {
            push(0, head);
        }
        for tail in [3u32, 1, 2, 1] {
            push(tail, 4);
        }

        assert_eq!(
            graph.node(NodeId(0)).successors(&graph),
            vec![NodeId(2), NodeId(1), NodeId(3)],
        );
        assert_eq!(
            graph.node(NodeId(4)).predecessors(&graph),
            vec![NodeId(3), NodeId(1), NodeId(2)],
        );
    }

    /// `superbubble_from` always terminates, returning `None`, on cyclic or back-edge adjacency —
    /// exercising the back-edge guard and the cyclic-termination path the DAG-only fixtures never
    /// reach. Fed directly (bypassing `Graph`, whose `topological_sort` asserts acyclicity).
    #[test]
    fn superbubble_from_terminates_on_cyclic_input() {
        // Back-edge straight to the entrance: 0 -> {2, 1}, 1 -> 0.
        let succ = vec![vec![2, 1], vec![0], vec![]];
        let pred = vec![vec![1], vec![0], vec![0]];
        assert_eq!(superbubble_from(&succ, &pred, 0), None);

        // A downstream 1<->2 cycle with a real sink: the mutually-dependent cycle nodes are never
        // settled (each waits on the other), so the walk drains and returns None rather than looping.
        let succ = vec![vec![1], vec![2], vec![1, 3], vec![]];
        let pred = vec![vec![], vec![0, 2], vec![1], vec![2]];
        assert_eq!(superbubble_from(&succ, &pred, 0), None);
    }

    /// The production Brankovic detector honors the documented cyclic-graph contract: on a cycle
    /// `reverse_post_order` fails, so `brankovic_superbubbles` reports nothing. (A real `Graph` is
    /// always a DAG, so this branch is fed adjacency directly.)
    #[test]
    fn brankovic_reports_nothing_on_a_cycle() {
        let succ = vec![vec![1], vec![0]]; // 0 -> 1 -> 0
        let pred = vec![vec![1], vec![0]];
        let exit_of = brankovic_superbubbles(&succ, &pred);
        assert!(exit_of.iter().all(Option::is_none));
    }

    /// A superbubble query on a graph built through the real `add_alignment` POA path, where a
    /// mismatch makes `rank_order()` diverge from node-id order (`0..n`). Exercises the
    /// `[source, rank_order]` emission on a non-identity rank order and confirms the Brankovic
    /// engine still equals both oracles there.
    #[test]
    fn superbubbles_on_real_poa_graph_with_diverging_rank_order() {
        let mut graph = Graph::new();
        // Seed A-C-G-T (nodes 0,1,2,3), then align A-T-G-T with a mismatch at column 1 (T vs C),
        // which adds node 4 (T) aligned to node 1; the aligned group makes rank_order reorder.
        graph.add_alignment(&[], b"ACGT", &[1, 1, 1, 1]).unwrap();
        graph
            .add_alignment(&[(0, 0), (1, 1), (2, 2), (3, 3)], b"ATGT", &[1, 1, 1, 1])
            .unwrap();

        let ranks: Vec<usize> = graph.rank_order().iter().map(|id| id.0 as usize).collect();
        assert_ne!(
            ranks,
            (0..graph.num_nodes()).collect::<Vec<_>>(),
            "fixture must exercise rank_order != node-id order"
        );
        assert_eq!(graph.superbubbles(), brute_force_superbubbles(&graph));
        assert_eq!(graph.superbubbles(), walk_superbubbles(&graph));
    }

    // ---- Differential property test against an independent brute-force oracle -----------------

    /// An independent, from-scratch smallest-superbubble detector, used only to differentially
    /// validate [`Graph::superbubbles`]. It rebuilds the augmented adjacency from the PUBLIC graph
    /// accessors (not [`Graph::augmented_adjacency`]) and, for each entrance in the same
    /// `[source, rank_order...]` order, returns the topologically-closest exit `t` for which
    /// `(entrance, t)` satisfies the formal superbubble condition:
    ///
    /// - `t` is reachable from `entrance`; and
    /// - over `U = { v : entrance ⇒* v and v ⇒* t }` (the vertices on some `entrance→t` path),
    ///   no edge escapes the region except at `t` (every `v ∈ U, v != t` has all successors in
    ///   `U`) and none enters except at `entrance` (every `v ∈ U, v != entrance` has all
    ///   predecessors in `U`).
    ///
    /// The closest valid exit is the smallest superbubble for that entrance, which is exactly what
    /// the Onodera–Sadakane walk reports — so the two must agree on every acyclic graph.
    fn brute_force_superbubbles(graph: &Graph) -> Vec<Superbubble> {
        let n = graph.num_nodes();
        let source = n;
        let sink = n + 1;
        let total = n + 2;

        // Independent augmented adjacency, from public accessors only.
        let mut succ: Vec<Vec<usize>> = vec![Vec::new(); total];
        let mut pred: Vec<Vec<usize>> = vec![Vec::new(); total];
        for i in 0..n {
            let node = graph.node(NodeId(i as u32));
            for &edge_id in &node.outedges {
                let head = graph.edge(edge_id).head.0 as usize;
                if !succ[i].contains(&head) {
                    succ[i].push(head);
                }
            }
            for &edge_id in &node.inedges {
                let tail = graph.edge(edge_id).tail.0 as usize;
                if !pred[i].contains(&tail) {
                    pred[i].push(tail);
                }
            }
        }
        for i in 0..n {
            let node = graph.node(NodeId(i as u32));
            if node.inedges.is_empty() {
                succ[source].push(i);
                pred[i].push(source);
            }
            if node.outedges.is_empty() {
                succ[i].push(sink);
                pred[sink].push(i);
            }
        }

        // Reachability closures over the augmented graph.
        let reachable = |adj: &[Vec<usize>], start: usize| -> Vec<bool> {
            let mut seen = vec![false; total];
            let mut stack = vec![start];
            seen[start] = true;
            while let Some(u) = stack.pop() {
                for &w in &adj[u] {
                    if !seen[w] {
                        seen[w] = true;
                        stack.push(w);
                    }
                }
            }
            seen
        };

        let is_superbubble = |s: usize, t: usize| -> bool {
            if s == t {
                return false;
            }
            let forward = reachable(&succ, s); // s ⇒* v
            if !forward[t] {
                return false; // t not reachable from s
            }
            let backward = reachable(&pred, t); // v ⇒* t
            let in_u = |v: usize| forward[v] && backward[v];
            for v in 0..total {
                if !in_u(v) {
                    continue;
                }
                if v != t && succ[v].iter().any(|&w| !in_u(w)) {
                    return false; // an edge escapes the region other than at t
                }
                if v != s && pred[v].iter().any(|&w| !in_u(w)) {
                    return false; // an edge enters the region other than at s
                }
            }
            true
        };

        // Topological position for ordering candidate exits: source < real nodes (rank order) <
        // sink. Every edge in the augmented graph goes strictly forward in this order.
        let mut topo_pos = vec![0usize; total];
        topo_pos[source] = 0;
        for (rank, id) in graph.rank_order().iter().enumerate() {
            topo_pos[id.0 as usize] = rank + 1;
        }
        topo_pos[sink] = total - 1;

        let smallest_from = |entrance: usize| -> Option<usize> {
            let forward = reachable(&succ, entrance);
            let mut candidates: Vec<usize> = (0..total)
                .filter(|&v| v != entrance && forward[v])
                .collect();
            candidates.sort_by_key(|&v| topo_pos[v]);
            candidates
                .into_iter()
                .find(|&t| is_superbubble(entrance, t))
        };

        std::iter::once(source)
            .chain(graph.rank_order().iter().map(|id| id.0 as usize))
            .filter_map(|entrance| {
                smallest_from(entrance).map(|exit| Superbubble {
                    entry: SuperbubbleEnd::from_augmented(entrance, n),
                    exit: SuperbubbleEnd::from_augmented(exit, n),
                })
            })
            .collect()
    }

    /// A strategy yielding random DAGs as `(node_count, edges)`, where every edge `(i, j)` has
    /// `i < j` (guaranteeing acyclicity). Each possible forward edge is independently included,
    /// so the space covers chains, diamonds, nested bubbles, and multi-source/multi-sink graphs.
    fn random_dag() -> impl Strategy<Value = (u32, Vec<(u32, u32)>)> {
        (1usize..=6).prop_flat_map(|n| {
            let forward_pairs: Vec<(u32, u32)> = (0..n)
                .flat_map(|i| ((i + 1)..n).map(move |j| (i as u32, j as u32)))
                .collect();
            let pair_count = forward_pairs.len();
            (
                Just(n as u32),
                proptest::collection::vec(any::<bool>(), pair_count).prop_map(move |mask| {
                    forward_pairs
                        .iter()
                        .zip(mask)
                        .filter_map(|(&pair, keep)| keep.then_some(pair))
                        .collect::<Vec<_>>()
                }),
            )
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]

        /// On every random DAG, the Brankovic `superbubbles()` returns exactly what the independent
        /// brute-force oracle AND the frontier-walk oracle compute — same bubbles, same order. This
        /// pins that the linear engine reproduces the earlier per-entrance walk with no change.
        #[test]
        fn superbubbles_matches_brute_force_oracle((n, edges) in random_dag()) {
            let graph = build(n, &edges);
            prop_assert_eq!(graph.superbubbles(), brute_force_superbubbles(&graph));
            prop_assert_eq!(graph.superbubbles(), walk_superbubbles(&graph));
        }

        /// Structural invariants of the raw output on every random DAG: repeated calls agree
        /// (determinism); entries are pairwise distinct (one bubble per entrance); `Source` only
        /// ever appears as an entry and `Sink` only ever as an exit, each at most once.
        #[test]
        fn superbubbles_structural_invariants((n, edges) in random_dag()) {
            let graph = build(n, &edges);
            let bubbles = graph.superbubbles();

            prop_assert_eq!(&bubbles, &graph.superbubbles());

            let mut entries: Vec<SuperbubbleEnd> = bubbles.iter().map(|b| b.entry).collect();
            let entry_count = entries.len();
            entries.sort_by_key(|e| match e {
                SuperbubbleEnd::Source => (0u8, 0u32),
                SuperbubbleEnd::Node(id) => (1, id.0),
                SuperbubbleEnd::Sink => (2, 0),
            });
            entries.dedup();
            prop_assert_eq!(entries.len(), entry_count, "entries must be pairwise distinct");

            let source_as_exit = bubbles.iter().filter(|b| b.exit == SuperbubbleEnd::Source).count();
            let sink_as_entry = bubbles.iter().filter(|b| b.entry == SuperbubbleEnd::Sink).count();
            prop_assert_eq!(source_as_exit, 0, "Source is never an exit");
            prop_assert_eq!(sink_as_entry, 0, "Sink is never an entry");
            let source_entries =
                bubbles.iter().filter(|b| b.entry == SuperbubbleEnd::Source).count();
            let sink_exits = bubbles.iter().filter(|b| b.exit == SuperbubbleEnd::Sink).count();
            prop_assert!(source_entries <= 1, "at most one Source-entry bubble");
            prop_assert!(sink_exits <= 1, "at most one Sink-exit bubble");
        }
    }
}
