//! Arena-based partial order alignment (POA) graph types.
//!
//! This module ports `spoa`'s pointer-based graph (`unique_ptr<Node>` / raw `Node*`) to an
//! index-based arena: nodes and edges live in flat `Vec`s on [`Graph`], and are referenced by
//! [`NodeId`] / [`EdgeId`] indices rather than pointers. This avoids `unsafe` entirely while
//! preserving the original's semantics.

use std::collections::HashSet;
use std::fmt;

/// Errors raised by [`Graph::add_alignment`] and its convenience wrappers.
///
/// Mirrors the three `std::invalid_argument` throws in `spoa::Graph::AddAlignment`
/// (`graph.cpp:162-166,184-188,191-194`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphError {
    /// `seq` and `weights` have different lengths (`graph.cpp:162-166`).
    UnequalWeights,
    /// An alignment entry's sequence index is out of bounds for `seq` (`graph.cpp:184-188`).
    InvalidAlignment,
    /// No alignment entry references any position in `seq` (`graph.cpp:191-194`).
    MissingSequence,
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            GraphError::UnequalWeights => "sequence and weights are of unequal size",
            GraphError::InvalidAlignment => "invalid alignment",
            GraphError::MissingSequence => "missing sequence in alignment",
        };
        write!(f, "[spoars::Graph::add_alignment] error: {message}")
    }
}

impl std::error::Error for GraphError {}

/// Index of a [`Node`] within [`Graph::nodes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

/// Index of an [`Edge`] within [`Graph::edges`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId(pub u32);

/// A single POA graph node: one aligned "column" position holding one coded symbol.
///
/// Mirrors `spoa::Graph::Node` (`graph.hpp:40-71`), minus the `id` field (an arena `Node`'s
/// identity is its index in [`Graph::nodes`], carried externally as a [`NodeId`]).
#[derive(Debug, Clone)]
pub struct Node {
    /// The coded symbol (see [`Graph::coder`] / [`Graph::decoder`]) this node represents.
    pub code: u32,
    /// Edges whose head is this node.
    pub inedges: Vec<EdgeId>,
    /// Edges whose tail is this node.
    pub outedges: Vec<EdgeId>,
    /// Other nodes aligned to this one (same column, different symbol).
    pub aligned_nodes: Vec<NodeId>,
}

impl Node {
    /// The distinct input-sequence labels passing through this node — the set-union of the
    /// sequence-index labels on all of its in-edges and out-edges — sorted ascending.
    ///
    /// This is the set [`Node::coverage`] counts; exposing it lets callers derive per-node strand
    /// or sequence membership without re-walking edges. See [`Graph::sequence_starts`] /
    /// [`Graph::sequence_path`] for what a label indexes.
    pub fn labels(&self, graph: &Graph) -> Vec<u32> {
        let mut labels: HashSet<u32> = HashSet::new();
        for &edge_id in self.inedges.iter().chain(self.outedges.iter()) {
            labels.extend(graph.edges[edge_id.0 as usize].labels.iter().copied());
        }
        let mut labels: Vec<u32> = labels.into_iter().collect();
        labels.sort_unstable();
        labels
    }

    /// Number of distinct input sequences passing through this node.
    ///
    /// Mirrors `spoa::Graph::Node::Coverage` (`graph.cpp:32-47`): the size of the set-union of
    /// the sequence-index labels on all of this node's in-edges and out-edges (i.e.
    /// `self.labels(graph).len()`).
    pub fn coverage(&self, graph: &Graph) -> u32 {
        self.labels(graph).len() as u32
    }

    /// Returns the head of this node's first out-edge (in insertion order) whose `labels`
    /// contains `label` — i.e. the next node visited by sequence `label` after this one — or
    /// `None` if no out-edge carries that label (this node is `label`'s last).
    ///
    /// Mirrors `spoa::Graph::Node::Successor` (`graph.cpp:22-30`).
    pub fn successor(&self, graph: &Graph, label: u32) -> Option<NodeId> {
        for &edge_id in &self.outedges {
            let edge = &graph.edges[edge_id.0 as usize];
            if edge.labels.contains(&label) {
                return Some(edge.head);
            }
        }
        None
    }

    /// The raw input byte this node represents, decoded via `graph` (`graph.decode(self.code)`),
    /// or `None` if the code is unknown to `graph` (which cannot happen for a node obtained from
    /// that same graph). A convenience over reading [`Node::code`] and calling [`Graph::decode`].
    pub fn base(&self, graph: &Graph) -> Option<u8> {
        graph.decode(self.code)
    }
}

/// A directed edge between two POA graph nodes, tagged with the sequences that traverse it.
///
/// Mirrors `spoa::Graph::Edge` (`graph.hpp:72-100`).
#[derive(Debug, Clone)]
pub struct Edge {
    /// Source node of this edge.
    pub tail: NodeId,
    /// Destination node of this edge.
    pub head: NodeId,
    /// Sequence-index labels of every input sequence that traverses this edge. Load-bearing for
    /// `Successor`-style MSA-row and GFA P-line reconstruction in later tasks.
    pub labels: Vec<u32>,
    /// Sum of per-sequence weights of all sequences traversing this edge.
    pub weight: i64,
}

/// An arena-based partial order alignment graph.
///
/// Mirrors `spoa::Graph` (`graph.hpp:25-320`), replacing its `unique_ptr<Node>` / raw `Node*`
/// pointer graph with flat `Vec<Node>` / `Vec<Edge>` arenas indexed by [`NodeId`] / [`EdgeId`].
#[derive(Debug, Clone)]
pub struct Graph {
    /// Arena of all nodes ever added to the graph, indexed by [`NodeId`].
    pub(crate) nodes: Vec<Node>,
    /// Arena of all edges ever added to the graph, indexed by [`EdgeId`].
    pub(crate) edges: Vec<Edge>,
    /// Maps a raw input byte (0-255) to its assigned code, or -1 if not yet seen.
    pub(crate) coder: [i32; 256],
    /// Maps a code back to its raw input byte, or -1 if the code is unused.
    pub(crate) decoder: Vec<i32>,
    /// Number of distinct symbol codes assigned so far.
    pub(crate) num_codes: u32,
    /// For each added sequence (in order), the [`NodeId`] of its first node.
    pub(crate) sequences: Vec<NodeId>,
    /// Nodes in topological-sort order, recomputed by [`Graph::topological_sort`] after every
    /// `add_alignment` call.
    pub(crate) rank_to_node: Vec<NodeId>,
    /// Nodes forming the generated consensus sequence, in traversal order, populated by
    /// [`Graph::traverse_heaviest_bundle`].
    pub(crate) consensus: Vec<NodeId>,
}

impl Default for Graph {
    fn default() -> Graph {
        Graph::new()
    }
}

impl Graph {
    /// Creates an empty graph, with the byte-to-code table cleared to "unassigned" (`-1`).
    ///
    /// Mirrors `spoa::Graph::Graph` (`graph.cpp:65-74`).
    pub fn new() -> Graph {
        Graph {
            nodes: Vec::new(),
            edges: Vec::new(),
            coder: [-1; 256],
            decoder: vec![-1; 256],
            num_codes: 0,
            sequences: Vec::new(),
            rank_to_node: Vec::new(),
            consensus: Vec::new(),
        }
    }

    // ---- Public read-only accessors -----------------------------------------------------------
    //
    // The graph is *built* through `add_alignment*` and *summarized* through `generate_consensus*`/
    // `generate_msa`/`to_gfa`/`to_dot`. These accessors expose the underlying arena so downstream
    // crates can also *inspect* the DAG directly — enumerate nodes/edges, follow the topological
    // order, read the consensus path as node ids, and decode a node's coded symbol back to its raw
    // input byte. All are immutable borrows: the arena is only ever mutated through the crate's own
    // graph-construction methods, preserving its invariants.

    /// All nodes ever added, indexed by [`NodeId`]'s inner `u32` (`self.nodes()[id.0 as usize]`,
    /// or use [`Graph::node`]). Nodes are never removed, so a [`NodeId`] stays valid for the
    /// graph's lifetime.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// All edges ever added, indexed by [`EdgeId`]'s inner `u32` (or use [`Graph::edge`]).
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// The [`Node`] for `id`.
    ///
    /// # Panics
    /// Panics if `id` is out of range (mirrors slice indexing); every id handed out by this crate
    /// is always in range.
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0 as usize]
    }

    /// The [`Edge`] for `id`.
    ///
    /// # Panics
    /// Panics if `id` is out of range (mirrors slice indexing).
    pub fn edge(&self, id: EdgeId) -> &Edge {
        &self.edges[id.0 as usize]
    }

    /// Number of nodes in the graph.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges in the graph.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Number of distinct symbol codes assigned so far (the alphabet size seen across all added
    /// sequences).
    pub fn num_codes(&self) -> u32 {
        self.num_codes
    }

    /// The raw input byte that a node's [`Node::code`] represents, or `None` if `code` was never
    /// assigned. Inverse of [`Graph::encode`]; use it to turn a node back into its base, e.g.
    /// `graph.decode(graph.node(id).code)`.
    pub fn decode(&self, code: u32) -> Option<u8> {
        match self.decoder.get(code as usize).copied() {
            Some(byte) if byte >= 0 => Some(byte as u8),
            _ => None,
        }
    }

    /// The code assigned to a raw input `base`, or `None` if that byte has not been seen in any
    /// added sequence. Inverse of [`Graph::decode`].
    pub fn encode(&self, base: u8) -> Option<u32> {
        match self.coder[base as usize] {
            code if code >= 0 => Some(code as u32),
            _ => None,
        }
    }

    /// Nodes in topological-sort order, recomputed after every `add_alignment*` call. Iterating
    /// this yields a valid processing order (every node appears after all of its predecessors);
    /// it is the same order the alignment fill and consensus traversal use.
    pub fn rank_order(&self) -> &[NodeId] {
        &self.rank_to_node
    }

    /// For each sequence added (in insertion order), the [`NodeId`] of its first node — the entry
    /// points for per-sequence traversal via [`Node::successor`].
    pub fn sequence_starts(&self) -> &[NodeId] {
        &self.sequences
    }

    /// The nodes of the most recently generated consensus path, in traversal order, or an empty
    /// slice if no consensus has been generated yet. Populated by [`Graph::generate_consensus`] /
    /// [`Graph::generate_consensus_min_coverage`]; `graph.consensus_nodes().iter().filter_map(|&n|
    /// graph.decode(graph.node(n).code))` reconstructs the same bytes those methods return as a
    /// `String`.
    pub fn consensus_nodes(&self) -> &[NodeId] {
        &self.consensus
    }

    /// The nodes sequence `seq_index` traverses, in order, from its start
    /// ([`Graph::sequence_starts`]`()[seq_index]`) following that sequence's labeled edges via
    /// [`Node::successor`], as a borrowing iterator (no allocation).
    ///
    /// `seq_index` is the 0-based index assigned in `add_alignment*` order (see the ordering
    /// guarantee on [`Graph::add_alignment`]); it is also the row index in [`Graph::generate_msa`].
    /// Decoding the visited nodes (`graph.decode(graph.node(id).code)`) reconstructs the sequence's
    /// bases in order.
    ///
    /// # Panics
    /// Panics if `seq_index >= self.sequence_starts().len()`.
    pub fn sequence_path_iter(&self, seq_index: usize) -> impl Iterator<Item = NodeId> + '_ {
        let label = seq_index as u32;
        let mut next = Some(self.sequences[seq_index]);
        std::iter::from_fn(move || {
            let current = next?;
            next = self.nodes[current.0 as usize].successor(self, label);
            Some(current)
        })
    }

    /// The nodes sequence `seq_index` traverses, in order, collected into a `Vec`. Allocating form
    /// of [`Graph::sequence_path_iter`]; see it for the index semantics.
    ///
    /// # Panics
    /// Panics if `seq_index >= self.sequence_starts().len()`.
    pub fn sequence_path(&self, seq_index: usize) -> Vec<NodeId> {
        self.sequence_path_iter(seq_index).collect()
    }

    /// For each [`NodeId`], the MSA column it occupies (aligned peer nodes share a column), plus the
    /// total column count. This is the exact mapping [`Graph::generate_msa`] uses internally to
    /// place each node's symbol; a downstream caller can use it to map bases to MSA columns without
    /// materializing the character-grid MSA.
    ///
    /// The returned `Vec` is indexed by `NodeId`'s inner `u32`; the `u32` is the number of columns.
    pub fn msa_columns(&self) -> (Vec<u32>, u32) {
        self.initialize_msa()
    }

    /// One entry per MSA column, listing the `(sequence_index, NodeId)` pairs present in that
    /// column — the column-major inverse of [`Graph::msa_columns`], saving callers from inverting
    /// the node→column map themselves.
    ///
    /// A `(sequence_index, node)` pair is emitted for column `c` when `sequence_index` traverses
    /// `node` (via [`Graph::sequence_path_iter`]) and `node` occupies column `c`. Within each
    /// column, pairs are ordered by `sequence_index` then by traversal order.
    pub fn column_members(&self) -> Vec<Vec<(u32, NodeId)>> {
        let (node_id_to_column, row_size) = self.initialize_msa();
        let mut columns: Vec<Vec<(u32, NodeId)>> = vec![Vec::new(); row_size as usize];
        for seq_index in 0..self.sequences.len() {
            for node in self.sequence_path_iter(seq_index) {
                let column = node_id_to_column[node.0 as usize] as usize;
                columns[column].push((seq_index as u32, node));
            }
        }
        columns
    }

    /// Extract the node-induced sub-DAG spanning parent node ids `begin..=end`.
    ///
    /// Returns the subgraph and a `Vec` indexed by the subgraph's `NodeId.0`, giving the
    /// corresponding `NodeId` in `self`. Mirrors `spoa::Graph::Subgraph` (`graph.cpp:574-628`),
    /// which internally calls `ExtractSubgraph(nodes_[end], nodes_[begin])` (`graph.cpp:551-571`).
    ///
    /// Node selection walks **backwards** from `end` over in-edges and aligned nodes, keeping any
    /// node whose id is `>= begin`. `num_codes` / `coder` / `decoder` are copied; in-edges are
    /// rebuilt (with the original weights) and aligned-node groups re-linked among kept nodes only;
    /// then the subgraph is topologically sorted. Per-sequence paths are **not** copied (spoa leaves
    /// `sequences_` empty — its `TODO(rvaser)`), so every subgraph edge is labeled `0` and each
    /// connected node reports `coverage() == 1`; this matches spoa exactly.
    ///
    /// # Panics
    /// Panics if `begin` or `end` is out of range (mirrors spoa indexing `nodes_[...]`).
    pub fn subgraph(&self, begin: NodeId, end: NodeId) -> (Graph, Vec<NodeId>) {
        // ExtractSubgraph: stack-DFS backwards from `end`, keeping nodes with id >= begin.
        let mut is_in_subgraph = vec![false; self.nodes.len()];
        let mut stack: Vec<NodeId> = vec![end];
        while let Some(curr) = stack.pop() {
            let idx = curr.0 as usize;
            if !is_in_subgraph[idx] && curr.0 >= begin.0 {
                for &edge_id in &self.nodes[idx].inedges {
                    stack.push(self.edges[edge_id.0 as usize].tail);
                }
                for &aligned in &self.nodes[idx].aligned_nodes {
                    stack.push(aligned);
                }
                is_in_subgraph[idx] = true;
            }
        }

        let mut subgraph = Graph::new();
        subgraph.num_codes = self.num_codes;
        subgraph.coder = self.coder;
        subgraph.decoder = self.decoder.clone();

        // subgraph_to_graph is indexed by subgraph node id; graph_to_subgraph inverts it.
        let mut subgraph_to_graph: Vec<NodeId> = Vec::new();
        let mut graph_to_subgraph: Vec<Option<NodeId>> = vec![None; self.nodes.len()];

        // Add nodes in parent id order (matches spoa's `for (const auto& it : nodes_)`).
        for (parent_idx, node) in self.nodes.iter().enumerate() {
            if !is_in_subgraph[parent_idx] {
                continue;
            }
            let new_id = subgraph.add_node(node.code);
            graph_to_subgraph[parent_idx] = Some(new_id);
            subgraph_to_graph.push(NodeId(parent_idx as u32));
        }

        // Connect: in-edges (with original weight) + aligned groups among kept nodes only.
        for (parent_idx, node) in self.nodes.iter().enumerate() {
            let Some(jt) = graph_to_subgraph[parent_idx] else {
                continue;
            };
            for &edge_id in &node.inedges {
                let edge = &self.edges[edge_id.0 as usize];
                if let Some(sub_tail) = graph_to_subgraph[edge.tail.0 as usize] {
                    // label = subgraph.sequences.len() (== 0), matching spoa's AddEdge.
                    let label = subgraph.sequences.len() as u32;
                    subgraph.add_edge(sub_tail, jt, edge.weight, label);
                }
            }
            for &aligned in &node.aligned_nodes {
                if let Some(sub_aligned) = graph_to_subgraph[aligned.0 as usize] {
                    subgraph.nodes[jt.0 as usize]
                        .aligned_nodes
                        .push(sub_aligned);
                }
            }
        }

        subgraph.topological_sort();
        (subgraph, subgraph_to_graph)
    }

    /// Rewrite an alignment computed against a subgraph back into this graph's node-id space.
    /// Maps only the node-id element (`.0`) of each `(node_id, seq_pos)` pair, via
    /// `subgraph_to_graph`; gap entries (`node_id == -1`) pass through unchanged. Mirrors
    /// `spoa::Graph::UpdateAlignment` (`graph.cpp:630-638`).
    ///
    /// # Panics
    /// Panics if a non-gap node id is out of range for `subgraph_to_graph`.
    pub fn update_alignment(
        &self,
        subgraph_to_graph: &[NodeId],
        alignment: &mut crate::align::Alignment,
    ) {
        for entry in alignment.iter_mut() {
            if entry.0 != -1 {
                entry.0 = subgraph_to_graph[entry.0 as usize].0 as i32;
            }
        }
    }

    /// Appends a new node with the given code and returns its id.
    ///
    /// Mirrors `spoa::Graph::AddNode` (`graph.cpp:76-79`): the new node's id is simply the
    /// arena's length before the push.
    pub(crate) fn add_node(&mut self, code: u32) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(Node {
            code,
            inedges: Vec::new(),
            outedges: Vec::new(),
            aligned_nodes: Vec::new(),
        });
        id
    }

    /// Adds a directed edge from `tail` to `head`, or if one already exists, folds the new
    /// sequence into it.
    ///
    /// Mirrors `spoa::Graph::AddEdge` (`graph.cpp:81-91`). `label` is the in-progress sequence's
    /// index (`sequences.len()` at the caller's call time); it is load-bearing for
    /// `Successor`-style MSA-row and GFA P-line reconstruction, so both the dedup path and the
    /// new-edge path record it.
    pub(crate) fn add_edge(&mut self, tail: NodeId, head: NodeId, weight: i64, label: u32) {
        for &edge_id in &self.nodes[tail.0 as usize].outedges {
            let edge = &mut self.edges[edge_id.0 as usize];
            if edge.head == head {
                edge.labels.push(label);
                edge.weight += weight;
                return;
            }
        }

        let edge_id = EdgeId(self.edges.len() as u32);
        self.edges.push(Edge {
            tail,
            head,
            labels: vec![label],
            weight,
        });
        self.nodes[tail.0 as usize].outedges.push(edge_id);
        self.nodes[head.0 as usize].inedges.push(edge_id);
    }

    /// Appends one node per base of `seq[begin..end]`, chained by edges weighted with the sum of
    /// each pair of adjacent bases' weights. Returns the id of the first node appended, or `None`
    /// if `begin == end`.
    ///
    /// Mirrors `spoa::Graph::AddSequence` (`graph.cpp:93-110`). Assumes every byte in
    /// `seq[begin..end]` has already been registered in [`Graph::coder`].
    fn add_sequence(
        &mut self,
        seq: &[u8],
        weights: &[u32],
        begin: u32,
        end: u32,
    ) -> Option<NodeId> {
        if begin == end {
            return None;
        }

        let label = self.sequences.len() as u32;
        let first = NodeId(self.nodes.len() as u32);
        let mut prev: Option<NodeId> = None;
        for i in begin..end {
            let code = self.coder[seq[i as usize] as usize] as u32;
            let curr = self.add_node(code);
            if let Some(p) = prev {
                let weight = weights[(i - 1) as usize] as i64 + weights[i as usize] as i64;
                self.add_edge(p, curr, weight, label);
            }
            prev = Some(curr);
        }
        Some(first)
    }

    /// Threads `seq` into the graph along `alignment`, its alignment to the graph's existing
    /// nodes, weighting each base-to-base edge by `weights`.
    ///
    /// `alignment` is a list of `(node_index, seq_index)` pairs mirroring `spoa::Alignment`
    /// (`vector<pair<int32_t, int32_t>>`): `-1` in either slot is the "no match" sentinel — a
    /// `-1` node index means the sequence base has no counterpart yet in the graph (insertion),
    /// and a `-1` sequence index means an existing graph node has no counterpart in this sequence
    /// (deletion).
    ///
    /// Mirrors `spoa::Graph::AddAlignment` (`graph.cpp:155-247`).
    ///
    /// # Sequence-index ordering (guaranteed)
    /// The Nth call to any `add_alignment*` method (`N` counting from 1, over non-empty sequences)
    /// assigns that sequence the label / index `N - 1`. That same 0-based index is:
    /// - the sequence's position in [`Graph::sequence_starts`],
    /// - the argument to [`Graph::sequence_path`] / [`Graph::sequence_path_iter`], and
    /// - the row index of that sequence in [`Graph::generate_msa`]'s output.
    ///
    /// This correspondence is a stable part of the API, not an implementation accident.
    pub fn add_alignment(
        &mut self,
        alignment: &[(i32, i32)],
        seq: &[u8],
        weights: &[u32],
    ) -> Result<(), GraphError> {
        if seq.is_empty() {
            return Ok(());
        }
        if seq.len() != weights.len() {
            return Err(GraphError::UnequalWeights);
        }

        // Coder registration (graph.cpp:168-173): assign every not-yet-seen byte a fresh code.
        for &base in seq {
            let byte = base as usize;
            if self.coder[byte] == -1 {
                self.coder[byte] = self.num_codes as i32;
                self.decoder[self.num_codes as usize] = base as i32;
                self.num_codes += 1;
            }
        }

        if alignment.is_empty() {
            // Empty-alignment fast path (graph.cpp:175-179): the whole sequence is new.
            if let Some(start) = self.add_sequence(seq, weights, 0, seq.len() as u32) {
                self.sequences.push(start);
            }
            self.topological_sort();
            return Ok(());
        }

        let mut valid: Vec<u32> = Vec::new();
        for &(node_idx, seq_idx) in alignment {
            if seq_idx != -1 {
                if seq_idx < 0 || seq_idx as usize >= seq.len() {
                    return Err(GraphError::InvalidAlignment);
                }
                // `node_idx` flows straight into `self.nodes[..]` in the aligned-bases loop
                // below; `-1` is the "new node" sentinel, so reject any other out-of-range
                // value here rather than let it panic on the index.
                if node_idx != -1 && (node_idx < 0 || node_idx as usize >= self.nodes.len()) {
                    return Err(GraphError::InvalidAlignment);
                }
                valid.push(seq_idx as u32);
            }
        }
        if valid.is_empty() {
            return Err(GraphError::MissingSequence);
        }
        let valid_front = valid[0];
        let valid_back = valid[valid.len() - 1];

        // Add unaligned prefix bases (positions before the first aligned base).
        let mut begin = self.add_sequence(seq, weights, 0, valid_front);
        let mut prev: Option<NodeId> = begin.map(|_| NodeId(self.nodes.len() as u32 - 1));
        // Add unaligned suffix bases (positions after the last aligned base) up front, matching
        // the C++ call order; `last` is the first node of that suffix chain, chained in below.
        let last = self.add_sequence(seq, weights, valid_back + 1, seq.len() as u32);

        // Add aligned bases (graph.cpp:202-240).
        for &(node_idx, seq_idx) in alignment {
            if seq_idx == -1 {
                continue;
            }
            let seq_idx = seq_idx as usize;
            let code = self.coder[seq[seq_idx] as usize] as u32;

            let curr = if node_idx == -1 {
                self.add_node(code)
            } else {
                let existing = NodeId(node_idx as u32);
                if self.nodes[existing.0 as usize].code == code {
                    existing
                } else {
                    let mut matched: Option<NodeId> = None;
                    for &aligned in &self.nodes[existing.0 as usize].aligned_nodes {
                        if self.nodes[aligned.0 as usize].code == code {
                            matched = Some(aligned);
                            break;
                        }
                    }
                    match matched {
                        Some(node) => node,
                        None => {
                            // Cross-link the new node into `existing`'s aligned-node set
                            // (graph.cpp:216-231).
                            let new_node = self.add_node(code);
                            let existing_aligned =
                                self.nodes[existing.0 as usize].aligned_nodes.clone();
                            for aligned in existing_aligned {
                                self.nodes[aligned.0 as usize].aligned_nodes.push(new_node);
                                self.nodes[new_node.0 as usize].aligned_nodes.push(aligned);
                            }
                            self.nodes[existing.0 as usize].aligned_nodes.push(new_node);
                            self.nodes[new_node.0 as usize].aligned_nodes.push(existing);
                            new_node
                        }
                    }
                }
            };

            if begin.is_none() {
                begin = Some(curr);
            }
            if let Some(p) = prev {
                let label = self.sequences.len() as u32;
                let weight = weights[seq_idx - 1] as i64 + weights[seq_idx] as i64;
                self.add_edge(p, curr, weight, label);
            }
            prev = Some(curr);
        }

        if let Some(last_node) = last {
            let label = self.sequences.len() as u32;
            let weight =
                weights[valid_back as usize] as i64 + weights[(valid_back + 1) as usize] as i64;
            let p = prev.expect("aligned-bases loop always sets prev when `valid` is non-empty");
            self.add_edge(p, last_node, weight, label);
        }
        if let Some(start) = begin {
            self.sequences.push(start);
        }

        self.topological_sort();
        Ok(())
    }

    /// Convenience wrapper: builds a uniform per-base `weights` vector from a single `weight` and
    /// delegates to [`Graph::add_alignment`].
    ///
    /// Mirrors `spoa::Graph::AddAlignment(alignment, sequence, sequence_len, weight)`
    /// (`graph.cpp:119-125`).
    pub fn add_alignment_weight(
        &mut self,
        alignment: &[(i32, i32)],
        seq: &[u8],
        weight: u32,
    ) -> Result<(), GraphError> {
        let weights = vec![weight; seq.len()];
        self.add_alignment(alignment, seq, &weights)
    }

    /// Convenience wrapper: derives per-base weights from Phred-scaled `quality` bytes
    /// (`quality[i] - 33`) and delegates to [`Graph::add_alignment`].
    ///
    /// Mirrors `spoa::Graph::AddAlignment(alignment, sequence, sequence_len, quality, quality_len)`
    /// (`graph.cpp:137-146`).
    pub fn add_alignment_quality(
        &mut self,
        alignment: &[(i32, i32)],
        seq: &[u8],
        quality: &[u8],
    ) -> Result<(), GraphError> {
        // Treats each quality byte as unsigned (0-255). C++'s `char` is signed on our targets, so
        // a byte >= 128 would produce a different (negative) `q - 33` before its narrowing to
        // uint32_t and thus diverge from the oracle. This is unreachable for valid Phred+33
        // (always < 127); a future differential-fuzz test feeding garbage quality bytes would need
        // to account for the sign difference here.
        let weights: Vec<u32> = quality.iter().map(|&q| (q as i32 - 33) as u32).collect();
        self.add_alignment(alignment, seq, &weights)
    }

    /// Computes a topological order of [`Graph::nodes`] into [`Graph::rank_to_node`].
    ///
    /// Mirrors `spoa::Graph::TopologicalSort` (`graph.cpp:249-303`): an iterative (explicit-stack)
    /// depth-first search over the graph, keyed by a per-node three-state `marks` array
    /// (0 = unvisited, 1 = on-stack/in-progress, 2 = done/ranked) and a per-node `ignored` flag.
    /// `ignored[n]` is set the moment `n` is discovered as another node's `aligned_nodes` entry;
    /// an ignored node never emits its own rank, since its rank is instead emitted immediately
    /// after the node that discovered it (see the `is_valid` branch below), keeping every
    /// aligned-node group contiguous in `rank_to_node`.
    ///
    /// For each node visited, unfinished (`marks != 2`) in-edge tails are pushed first, then —
    /// only if the node itself is not ignored — unfinished aligned nodes are pushed (and marked
    /// ignored). This exact push order is load-bearing: it determines the resulting rank order,
    /// which later tasks' DP indexing and consensus tie-breaking depend on matching byte-for-byte.
    fn topological_sort(&mut self) {
        self.rank_to_node.clear();

        let mut marks = vec![0u8; self.nodes.len()];
        let mut ignored = vec![false; self.nodes.len()];
        let mut stack: Vec<NodeId> = Vec::new();

        for start in 0..self.nodes.len() {
            if marks[start] != 0 {
                continue;
            }
            stack.push(NodeId(start as u32));

            while let Some(&curr) = stack.last() {
                let curr_idx = curr.0 as usize;
                let mut is_valid = true;

                if marks[curr_idx] != 2 {
                    for &edge_id in &self.nodes[curr_idx].inedges {
                        let tail = self.edges[edge_id.0 as usize].tail;
                        if marks[tail.0 as usize] != 2 {
                            stack.push(tail);
                            is_valid = false;
                        }
                    }

                    if !ignored[curr_idx] {
                        for &aligned in &self.nodes[curr_idx].aligned_nodes {
                            if marks[aligned.0 as usize] != 2 {
                                stack.push(aligned);
                                ignored[aligned.0 as usize] = true;
                                is_valid = false;
                            }
                        }
                    }

                    debug_assert!(is_valid || marks[curr_idx] != 1, "Graph is not a DAG");

                    if is_valid {
                        marks[curr_idx] = 2;
                        if !ignored[curr_idx] {
                            self.rank_to_node.push(curr);
                            for &aligned in &self.nodes[curr_idx].aligned_nodes {
                                self.rank_to_node.push(aligned);
                            }
                        }
                    } else {
                        marks[curr_idx] = 1;
                    }
                }

                if is_valid {
                    stack.pop();
                }
            }
        }

        debug_assert!(
            self.is_topologically_sorted(),
            "Graph is not topologically sorted"
        );
    }

    /// Generates the consensus sequence, keeping every node regardless of coverage.
    ///
    /// Equivalent to `min_coverage = -1`: since [`Node::coverage`] is always `>= 0`, no node is
    /// ever excluded. Mirrors `spoa::Graph::GenerateConsensus()` (`graph.cpp:368-374`), which
    /// itself never filters by coverage.
    ///
    /// # Note on run-length-sensitive data
    /// The consensus is the graph's *heaviest bundle* — the path maximizing summed edge weight — so
    /// across long homopolymer or low-complexity runs the consensus length can be inflated relative
    /// to a per-column majority vote. This is correct spoa behavior. If you need run-length-faithful
    /// output, prefer the column view ([`Graph::msa_columns`] / [`Graph::column_members`] /
    /// [`Graph::generate_msa`]) and reduce per column yourself.
    pub fn generate_consensus(&mut self) -> String {
        self.generate_consensus_min_coverage(-1)
    }

    /// Generates the consensus sequence, dropping any consensus node whose [`Node::coverage`] is
    /// below `min_coverage`.
    ///
    /// Mirrors `spoa::Graph::GenerateConsensus(std::int32_t min_coverage)` (`graph.cpp:377-386`).
    pub fn generate_consensus_min_coverage(&mut self, min_coverage: i32) -> String {
        self.traverse_heaviest_bundle();

        let mut dst = String::new();
        for i in 0..self.consensus.len() {
            let node_id = self.consensus[i];
            let node = &self.nodes[node_id.0 as usize];
            if node.coverage(self) as i32 >= min_coverage {
                let code = node.code;
                let byte = self.decoder[code as usize];
                dst.push(byte as u8 as char);
            }
        }
        dst
    }

    /// Maps every node id to its multiple-sequence-alignment column, folding each node's
    /// `aligned_nodes` peers into the same column as their representative. Returns
    /// `(node_id_to_column, row_size)`, where `row_size` is the total number of distinct MSA
    /// columns (the number of aligned-node *groups* in topological order, not [`Graph::nodes`]'s
    /// length).
    ///
    /// Mirrors `spoa::Graph::InitializeMultipleSequenceAlignment` (`graph.cpp:321-337`). That
    /// loop's C-style `for (i = 0; i < rank_to_node_.size(); ++i, ++j)` advances `j` once per
    /// outer iteration, but its body *also* does `++i` once per aligned peer
    /// (`graph.cpp:325-331`) to skip over the peers that `topological_sort` already placed
    /// immediately after their representative in `rank_to_node`. A plain `for (i, node) in
    /// rank_to_node.iter().enumerate()` cannot replicate that mid-loop `++i`, so this is ported
    /// with an explicit `while` loop and a manually advanced `i`, matching both increments
    /// exactly (see the module-level task notes on this double-increment).
    fn initialize_msa(&self) -> (Vec<u32>, u32) {
        let mut node_id_to_column = vec![0u32; self.nodes.len()];
        let mut j: u32 = 0;
        let mut i: usize = 0;
        while i < self.rank_to_node.len() {
            let node_id = self.rank_to_node[i];
            node_id_to_column[node_id.0 as usize] = j;
            for &aligned in &self.nodes[node_id.0 as usize].aligned_nodes {
                node_id_to_column[aligned.0 as usize] = j;
                i += 1;
            }
            i += 1;
            j += 1;
        }
        (node_id_to_column, j)
    }

    /// Generates a multiple sequence alignment: one row per input sequence (in `add_alignment`
    /// order), each padded to the same width with `'-'` gap characters, plus an optional
    /// trailing consensus row.
    ///
    /// Mirrors `spoa::Graph::GenerateMultipleSequenceAlignment` (`graph.cpp:339-366`). Each
    /// sequence's row is built by walking its nodes via [`Node::successor`] (labeled by that
    /// sequence's index) from its first node to its last, writing each visited node's decoded
    /// symbol into that node's MSA column (from [`Graph::initialize_msa`]); columns the sequence
    /// never visits stay `'-'`. When `include_consensus` is `true`,
    /// [`Graph::traverse_heaviest_bundle`] is (re-)run and one more row is appended the same way,
    /// from [`Graph::consensus`].
    pub fn generate_msa(&mut self, include_consensus: bool) -> Vec<String> {
        let (node_id_to_column, row_size) = self.initialize_msa();

        let mut dst: Vec<String> = Vec::with_capacity(self.sequences.len() + 1);
        for i in 0..self.sequences.len() {
            let mut row = vec!['-'; row_size as usize];
            for node_id in self.sequence_path_iter(i) {
                let code = self.nodes[node_id.0 as usize].code;
                let byte = self.decoder[code as usize];
                row[node_id_to_column[node_id.0 as usize] as usize] = byte as u8 as char;
            }
            dst.push(row.into_iter().collect());
        }

        if include_consensus {
            self.traverse_heaviest_bundle();
            let mut row = vec!['-'; row_size as usize];
            for i in 0..self.consensus.len() {
                let node_id = self.consensus[i];
                let code = self.nodes[node_id.0 as usize].code;
                let byte = self.decoder[code as usize];
                row[node_id_to_column[node_id.0 as usize] as usize] = byte as u8 as char;
            }
            dst.push(row.into_iter().collect());
        }

        dst
    }

    /// Finds the heaviest-weighted path through the graph (the "heaviest bundle") and stores it,
    /// in traversal order, into [`Graph::consensus`].
    ///
    /// Mirrors `spoa::Graph::TraverseHeaviestBundle` (`graph.cpp:465-509`). Runs a single
    /// forward DP pass over [`Graph::rank_to_node`], scoring each node by the heaviest-weighted
    /// path reaching it (`scores`) and recording that path's predecessor (`predecessors`); ties
    /// are broken by preferring the inedge whose *tail's own* predecessor chain scored highest
    /// (see the tie-break note on the inedge comparison below). If the best-scoring node found
    /// (`max`) still has outgoing edges once the forward pass completes, the bundle has run into
    /// a competing branch partway through, so [`Graph::branch_completion`] is called repeatedly
    /// (invalidating and rescoring the losing branch) until `max` is a true sink. The consensus is
    /// then read off by walking `predecessors` back from `max` to the bundle's start and
    /// reversing.
    fn traverse_heaviest_bundle(&mut self) {
        if self.rank_to_node.is_empty() {
            return;
        }

        let mut predecessors: Vec<Option<NodeId>> = vec![None; self.nodes.len()];
        let mut scores: Vec<i64> = vec![-1; self.nodes.len()];
        let mut max: Option<NodeId> = None;

        for i in 0..self.rank_to_node.len() {
            let node_id = self.rank_to_node[i];
            let node_idx = node_id.0 as usize;
            for &edge_id in &self.nodes[node_idx].inedges {
                let edge = &self.edges[edge_id.0 as usize];
                // Tie-break short-circuit (graph.cpp:477-478): mirrors C++'s
                // `(scores[it->id] < jt->weight) || (scores[it->id] == jt->weight &&
                // scores[predecessors[it->id]->id] <= scores[jt->tail->id])` exactly. The `&&`
                // only evaluates its right side when the first `scores[it] < weight` comparison
                // is false; C++ relies on this to dereference `predecessors[it->id]` safely
                // (weights are positive and scores start at -1, so on a node's first inedge the
                // first comparison is always true and the null predecessor is never touched).
                // With `Option<NodeId>` we replicate that: only read `predecessors[node_idx]`
                // (and `.expect` it to be `Some`) inside the branch that C++'s short-circuit
                // guarantees is only reached once a predecessor already exists.
                let take = if scores[node_idx] < edge.weight {
                    true
                } else if scores[node_idx] == edge.weight {
                    let current_pred = predecessors[node_idx].expect(
                        "scores[node_idx] == edge.weight (both != -1's initial trivial case \
                         would have taken the first branch) implies a predecessor was already \
                         recorded",
                    );
                    scores[current_pred.0 as usize] <= scores[edge.tail.0 as usize]
                } else {
                    false
                };
                if take {
                    scores[node_idx] = edge.weight;
                    predecessors[node_idx] = Some(edge.tail);
                }
            }
            if let Some(pred) = predecessors[node_idx] {
                scores[node_idx] += scores[pred.0 as usize];
            }
            if max.is_none() || scores[max.unwrap().0 as usize] < scores[node_idx] {
                max = Some(node_id);
            }
        }

        let mut max = max.expect("rank_to_node is non-empty, so max is always assigned");

        if !self.nodes[max.0 as usize].outedges.is_empty() {
            let mut node_id_to_rank = vec![0u32; self.nodes.len()];
            for (rank, &node_id) in self.rank_to_node.iter().enumerate() {
                node_id_to_rank[node_id.0 as usize] = rank as u32;
            }
            while !self.nodes[max.0 as usize].outedges.is_empty() {
                let rank = node_id_to_rank[max.0 as usize];
                max = self.branch_completion(rank, &mut scores, &mut predecessors);
            }
        }

        // Traceback (graph.cpp:502-508).
        self.consensus.clear();
        let mut node = max;
        while let Some(pred) = predecessors[node.0 as usize] {
            self.consensus.push(node);
            node = pred;
        }
        self.consensus.push(node);
        self.consensus.reverse();
    }

    /// Resolves a heaviest-bundle traversal that ran into a competing branch at `rank`: the
    /// branch not taken by `rank`'s node is invalidated (scores reset to `-1`), then every node
    /// from `rank + 1` onward is rescored, skipping any inedge whose tail was invalidated (either
    /// just now, or earlier in this same rescan). Returns the best-scoring node found in the
    /// rescanned range.
    ///
    /// Mirrors `spoa::Graph::BranchCompletion` (`graph.cpp:511-549`). The inedge tie-break
    /// comparison is the same short-circuit as [`Graph::traverse_heaviest_bundle`]'s forward
    /// pass; see that method's doc comment for why the predecessor dereference is safe.
    fn branch_completion(
        &self,
        rank: u32,
        scores: &mut [i64],
        predecessors: &mut [Option<NodeId>],
    ) -> NodeId {
        let start = self.rank_to_node[rank as usize];
        for &out_edge_id in &self.nodes[start.0 as usize].outedges {
            let head = self.edges[out_edge_id.0 as usize].head;
            for &in_edge_id in &self.nodes[head.0 as usize].inedges {
                let tail = self.edges[in_edge_id.0 as usize].tail;
                if tail != start {
                    scores[tail.0 as usize] = -1;
                }
            }
        }

        let mut max: Option<NodeId> = None;
        for i in (rank as usize + 1)..self.rank_to_node.len() {
            let node_id = self.rank_to_node[i];
            let node_idx = node_id.0 as usize;
            scores[node_idx] = -1;
            predecessors[node_idx] = None;

            for &edge_id in &self.nodes[node_idx].inedges {
                let edge = &self.edges[edge_id.0 as usize];
                if scores[edge.tail.0 as usize] == -1 {
                    continue;
                }
                // Same tie-break short-circuit as traverse_heaviest_bundle (graph.cpp:534-535).
                let take = if scores[node_idx] < edge.weight {
                    true
                } else if scores[node_idx] == edge.weight {
                    let current_pred = predecessors[node_idx].expect(
                        "scores[node_idx] == edge.weight implies a predecessor was already \
                         recorded on an earlier (non-skipped) inedge",
                    );
                    scores[current_pred.0 as usize] <= scores[edge.tail.0 as usize]
                } else {
                    false
                };
                if take {
                    scores[node_idx] = edge.weight;
                    predecessors[node_idx] = Some(edge.tail);
                }
            }
            if let Some(pred) = predecessors[node_idx] {
                scores[node_idx] += scores[pred.0 as usize];
            }
            if max.is_none() || scores[max.unwrap().0 as usize] < scores[node_idx] {
                max = Some(node_id);
            }
        }

        max.expect("rank + 1 < rank_to_node.len() whenever BranchCompletion is called, since it's only called while `max` still has outedges")
    }

    /// Emits the graph in GFA (Graphical Fragment Assembly) format: one `H` header line, one
    /// `S`/`L` segment/link line per node/out-edge, and one `P` path line per input sequence
    /// (plus, when `include_consensus` is `true`, one final `P\tConsensus\t...` line).
    ///
    /// Mirrors `spoa::PrintGfa` (`third_party/spoa/src/main.cpp:123-203`), returning a `String`
    /// instead of writing to `std::cout`. `headers[i]` becomes sequence `i`'s `P`-line name;
    /// `is_reversed[i]`, if non-empty, reverses that same `P`-line's node path and flips every
    /// node's orientation suffix from `+` to `-`. Node and edge ids are emitted 1-based
    /// (`id + 1`), matching upstream's GFA convention (GFA disallows id `0`). A node (or a `L`
    /// line whose *both* endpoints) that appears in [`Graph::consensus`] gets a trailing
    /// `\tic:Z:true` tag; [`Graph::consensus`] must already be populated by a prior
    /// [`Graph::generate_consensus`] (or a min-coverage variant) call, since this method only
    /// reads it and never recomputes it.
    ///
    /// # Panics (debug only)
    ///
    /// Debug-asserts that `headers` has at least one entry per sequence, and that
    /// `is_reversed` (if non-empty) does too — mirroring upstream's
    /// `[spoa::PrintGfa] error: missing header(s)` early return, but as a precondition rather
    /// than a silent empty-string result, since this library's callers (the oracle, and later
    /// the CLI) always supply enough of both.
    pub fn to_gfa(
        &self,
        headers: &[String],
        is_reversed: &[bool],
        include_consensus: bool,
    ) -> String {
        debug_assert!(
            headers.len() >= self.sequences.len(),
            "to_gfa: missing header(s): {} header(s) for {} sequence(s)",
            headers.len(),
            self.sequences.len()
        );
        debug_assert!(
            is_reversed.is_empty() || is_reversed.len() >= self.sequences.len(),
            "to_gfa: missing reversion flag(s): {} flag(s) for {} sequence(s)",
            is_reversed.len(),
            self.sequences.len()
        );

        let mut is_consensus_node = vec![false; self.nodes.len()];
        for &node_id in &self.consensus {
            is_consensus_node[node_id.0 as usize] = true;
        }

        let mut out = String::new();
        out.push_str("H\tVN:Z:1.0\n");
        for (id, node) in self.nodes.iter().enumerate() {
            let symbol = self.decoder[node.code as usize] as u8 as char;
            out.push_str(&format!("S\t{}\t{symbol}", id + 1));
            if is_consensus_node[id] {
                out.push_str("\tic:Z:true");
            }
            out.push('\n');

            for &edge_id in &node.outedges {
                let edge = &self.edges[edge_id.0 as usize];
                let head_id = edge.head.0 as usize;
                out.push_str(&format!(
                    "L\t{}\t+\t{}\t+\tOM\tew:f:{}",
                    id + 1,
                    head_id + 1,
                    edge.weight
                ));
                if is_consensus_node[id] && is_consensus_node[head_id] {
                    out.push_str("\tic:Z:true");
                }
                out.push('\n');
            }
        }

        for i in 0..self.sequences.len() {
            out.push_str(&format!("P\t{}\t", headers[i]));

            // 1-based node ids along this sequence's path (see `sequence_path_iter`).
            let mut path: Vec<u32> = self.sequence_path_iter(i).map(|node| node.0 + 1).collect();

            let reversed = !is_reversed.is_empty() && is_reversed[i];
            if reversed {
                path.reverse();
            }
            let sign = if reversed { '-' } else { '+' };
            for (j, node_id) in path.iter().enumerate() {
                if j != 0 {
                    out.push(',');
                }
                out.push_str(&format!("{node_id}{sign}"));
            }
            out.push_str("\t*\n");
        }

        if include_consensus {
            out.push_str("P\tConsensus\t");
            for (i, &node_id) in self.consensus.iter().enumerate() {
                if i != 0 {
                    out.push(',');
                }
                out.push_str(&format!("{}+", node_id.0 + 1));
            }
            out.push_str("\t*\n");
        }

        out
    }

    /// Emits the graph in Graphviz DOT format.
    ///
    /// Mirrors `spoa::Graph::PrintDot` (`third_party/spoa/src/graph.cpp:640-680`), returning a
    /// `String` instead of writing to a file. Node and edge ids are emitted 0-based (unlike
    /// [`Graph::to_gfa`]'s 1-based ids). A node in [`Graph::consensus`] is filled `goldenrod1`;
    /// an edge between two CONSECUTIVE consensus nodes (`consensus_rank[tail] + 1 ==
    /// consensus_rank[head]`, `-1` for non-consensus nodes — reproduced verbatim, including its
    /// edge case where a non-consensus tail's `-1 + 1 == 0` can coincidentally match a
    /// consensus head at rank 0) is colored `goldenrod1`; and each aligned-node pair (same MSA
    /// column, different symbol) is linked once by a dotted, arrowhead-less edge, guarded by
    /// `jt.id > it.id` so each pair is only emitted from its lower-id member. [`Graph::consensus`]
    /// must already be populated by a prior [`Graph::generate_consensus`] (or a min-coverage
    /// variant) call, since this method only reads it and never recomputes it.
    pub fn to_dot(&self) -> String {
        let mut consensus_rank: Vec<i32> = vec![-1; self.nodes.len()];
        for (rank, &node_id) in self.consensus.iter().enumerate() {
            consensus_rank[node_id.0 as usize] = rank as i32;
        }

        let mut out = String::new();
        out.push_str(&format!(
            "digraph {} {{\n  graph [rankdir = LR]\n",
            self.sequences.len()
        ));

        for (id, node) in self.nodes.iter().enumerate() {
            let symbol = self.decoder[node.code as usize] as u8 as char;
            out.push_str(&format!("  {id}[label = \"{id} - {symbol}\""));
            if consensus_rank[id] != -1 {
                out.push_str(", style = filled, fillcolor = goldenrod1");
            }
            out.push_str("]\n");

            for &edge_id in &node.outedges {
                let edge = &self.edges[edge_id.0 as usize];
                let head_id = edge.head.0 as usize;
                out.push_str(&format!("  {id} -> {head_id} [label = \"{}\"", edge.weight));
                if consensus_rank[id] + 1 == consensus_rank[head_id] {
                    out.push_str(", color = goldenrod1");
                }
                out.push_str("]\n");
            }
            for &aligned in &node.aligned_nodes {
                let aligned_id = aligned.0 as usize;
                if aligned_id > id {
                    out.push_str(&format!(
                        "  {id} -> {aligned_id} [style = dotted, arrowhead = none]\n"
                    ));
                }
            }
        }
        out.push_str("}\n");

        out
    }

    /// Returns whether [`Graph::rank_to_node`] is a valid topological order of [`Graph::nodes`]:
    /// every node's in-edge tails appear at an earlier rank than the node itself.
    ///
    /// Mirrors `spoa::Graph::IsTopologicallySorted` (`graph.cpp:305-319`).
    pub fn is_topologically_sorted(&self) -> bool {
        debug_assert!(
            self.nodes.len() == self.rank_to_node.len(),
            "Topological sort not called"
        );

        let mut visited = vec![false; self.nodes.len()];
        for &node_id in &self.rank_to_node {
            let node = &self.nodes[node_id.0 as usize];
            for &edge_id in &node.inedges {
                let tail = self.edges[edge_id.0 as usize].tail;
                if !visited[tail.0 as usize] {
                    return false;
                }
            }
            visited[node_id.0 as usize] = true;
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::AlignmentEngine;
    use proptest::prelude::*;

    #[test]
    fn add_node_assigns_sequential_ids() {
        let mut g = Graph::new();
        let a = g.add_node(0);
        let b = g.add_node(1);
        assert_eq!(a.0, 0);
        assert_eq!(b.0, 1);
        assert_eq!(g.nodes.len(), 2);
    }

    /// The public read-only accessors expose a coherent view of a built graph: the arena slices
    /// and their id-indexed lookups agree, codes round-trip through `encode`/`decode`, and the
    /// rank/sequence/consensus views are populated and decodable.
    #[test]
    fn public_accessors_expose_a_coherent_view_of_a_built_graph() {
        let mut g = Graph::new();
        // First sequence seeds the linear chain (nodes 0,1,2); the second is aligned onto it
        // position-for-position so the two merge into one coverage-2 chain rather than a disjoint
        // second chain.
        g.add_alignment(&[], b"ACG", &[1, 1, 1]).unwrap();
        g.add_alignment(&[(0, 0), (1, 1), (2, 2)], b"ACG", &[1, 1, 1])
            .unwrap();

        // Arena slices and id lookups agree.
        assert_eq!(g.num_nodes(), g.nodes().len());
        assert_eq!(g.num_edges(), g.edges().len());
        assert_eq!(g.num_nodes(), 3);
        for (i, node) in g.nodes().iter().enumerate() {
            let id = NodeId(i as u32);
            assert_eq!(g.node(id).code, node.code);
        }
        for (i, edge) in g.edges().iter().enumerate() {
            let id = EdgeId(i as u32);
            assert_eq!(g.edge(id).tail, edge.tail);
        }

        // Codes round-trip; the alphabet is exactly {A, C, G}.
        assert_eq!(g.num_codes(), 3);
        for &base in b"ACG" {
            let code = g.encode(base).expect("base was added");
            assert_eq!(g.decode(code), Some(base));
        }
        assert_eq!(g.encode(b'T'), None); // never seen
        assert_eq!(g.decode(999), None); // never assigned

        // Node::base decodes each node back to its input byte, in rank order spelling "ACG".
        let spelled: Vec<u8> = g
            .rank_order()
            .iter()
            .map(|&id| g.node(id).base(&g).expect("node code is decodable"))
            .collect();
        assert_eq!(spelled, b"ACG");
        assert_eq!(g.rank_order().len(), 3);

        // One start per added sequence; every node has coverage 2 (both sequences traverse it).
        assert_eq!(g.sequence_starts().len(), 2);
        for node in g.nodes() {
            assert_eq!(node.coverage(&g), 2);
        }

        // Consensus is empty until generated, then decodes to the same String the API returns.
        assert!(g.consensus_nodes().is_empty());
        let consensus = g.generate_consensus();
        let from_nodes: String = g
            .consensus_nodes()
            .iter()
            .map(|&id| g.node(id).base(&g).expect("node code is decodable") as char)
            .collect();
        assert_eq!(from_nodes, consensus);
        assert_eq!(consensus, "ACG");
    }

    /// Builds a small graph with one aligned (substituted) column so the MSA has a folded column,
    /// and exercises the RELAY accessors: `sequence_path` decodes back to each input;
    /// `msa_columns`/`column_members` agree and fold aligned peers; `Node::labels` matches coverage.
    #[test]
    fn relay_accessors_map_sequences_to_msa_columns() {
        // "ACGT" then "ACTT" aligned onto it: position 2 diverges (G vs T) into an aligned column.
        let mut g = Graph::new();
        g.add_alignment(&[], b"ACGT", &[1, 1, 1, 1]).unwrap();
        g.add_alignment(&[(0, 0), (1, 1), (2, 2), (3, 3)], b"ACTT", &[1, 1, 1, 1])
            .unwrap();
        assert_eq!(g.sequence_starts().len(), 2);

        // sequence_path (and its iterator) decode back to the exact input bytes, indexed by the
        // add order (seq 0 = "ACGT", seq 1 = "ACTT").
        let decode_path = |g: &Graph, i: usize| -> Vec<u8> {
            g.sequence_path(i)
                .iter()
                .map(|&n| g.node(n).base(g).unwrap())
                .collect()
        };
        assert_eq!(decode_path(&g, 0), b"ACGT");
        assert_eq!(decode_path(&g, 1), b"ACTT");
        // The iterator form yields the same node sequence.
        assert_eq!(
            g.sequence_path_iter(0).collect::<Vec<_>>(),
            g.sequence_path(0)
        );

        // msa_columns: 4 columns; the divergent G and T share their column (aligned peers folded).
        let (node_to_col, row_size) = g.msa_columns();
        assert_eq!(row_size, 4);
        let path0 = g.sequence_path(0); // A C G T
        let path1 = g.sequence_path(1); // A C T T
        assert_eq!(
            node_to_col[path0[0].0 as usize],
            node_to_col[path1[0].0 as usize]
        ); // shared A
        assert_eq!(
            node_to_col[path0[2].0 as usize],
            node_to_col[path1[2].0 as usize]
        ); // G/T folded
        assert_ne!(path0[2], path1[2]); // ...but they are distinct nodes

        // column_members inverts the map: the divergent column lists both sequences; column 0 too.
        let columns = g.column_members();
        assert_eq!(columns.len(), row_size as usize);
        let divergent = node_to_col[path0[2].0 as usize] as usize;
        let seqs_in_divergent: Vec<u32> = columns[divergent].iter().map(|&(s, _)| s).collect();
        assert_eq!(seqs_in_divergent, vec![0, 1]);
        // Every (seq, node) entry actually sits in the column it is filed under.
        for (col, members) in columns.iter().enumerate() {
            for &(_seq, node) in members {
                assert_eq!(node_to_col[node.0 as usize] as usize, col);
            }
        }

        // Node::labels == the set coverage counts, sorted; the shared "A" node carries both labels.
        let a_node = path0[0];
        assert_eq!(g.node(a_node).labels(&g), vec![0, 1]);
        for node in g.nodes() {
            assert_eq!(node.labels(&g).len() as u32, node.coverage(&g));
        }
    }

    /// The MSA row order matches the `sequence_path`/`sequence_starts` index order (RELAY #3 doc
    /// guarantee), and each MSA row's non-gap characters decode that sequence.
    #[test]
    fn generate_msa_row_order_matches_sequence_index() {
        let mut g = Graph::new();
        g.add_alignment(&[], b"ACGT", &[1, 1, 1, 1]).unwrap();
        g.add_alignment(&[(0, 0), (1, 1), (2, 2), (3, 3)], b"ACTT", &[1, 1, 1, 1])
            .unwrap();
        let msa = g.generate_msa(false);
        assert_eq!(msa.len(), 2);
        for (i, row) in msa.iter().enumerate() {
            let ungapped: String = row.chars().filter(|&c| c != '-').collect();
            let expected: String = g
                .sequence_path(i)
                .iter()
                .map(|&n| g.node(n).base(&g).unwrap() as char)
                .collect();
            assert_eq!(ungapped, expected, "MSA row {i} must spell sequence {i}");
        }
    }

    #[test]
    fn add_edge_dedups_repeated_tail_head_pairs() {
        let mut g = Graph::new();
        let a = g.add_node(0);
        let b = g.add_node(1);

        g.add_edge(a, b, 1, 0);
        g.add_edge(a, b, 1, 1);

        assert_eq!(
            g.edges.len(),
            1,
            "second add_edge should reuse the existing edge"
        );
        let edge = &g.edges[0];
        assert_eq!(edge.labels, vec![0, 1]);
        assert_eq!(edge.weight, 2);
        assert_eq!(g.nodes[a.0 as usize].outedges, vec![EdgeId(0)]);
        assert_eq!(g.nodes[b.0 as usize].inedges, vec![EdgeId(0)]);
    }

    /// Locks the tie-break short-circuit in `traverse_heaviest_bundle`'s inedge comparison
    /// (mirroring `graph.cpp:477-478`) independent of the oracle. Builds a diamond
    /// `S -> A -> M` / `S -> B -> M` where `A->M` and `B->M` have EQUAL weight but `A`'s
    /// cumulative score (via `S->A`) is far heavier than `B`'s (via `S->B`), so the correct
    /// winner at the tie is decided by comparing `scores[predecessors[M]]` against
    /// `scores[jt.tail]` — not by the local edge weight (which is tied) and not by insertion
    /// order. Exercised with the two inedges added in BOTH orders to confirm the outcome (and
    /// the absence of any panic from an eager predecessor `.unwrap()`) is order-independent.
    #[test]
    fn traverse_heaviest_bundle_breaks_equal_weight_inedge_ties_by_predecessor_score() {
        for reversed in [false, true] {
            let mut g = Graph::new();
            let s = g.add_node(0);
            let a = g.add_node(1);
            let b = g.add_node(2);
            let m = g.add_node(3);
            // Decoder entries so `generate_consensus` produces readable output.
            g.decoder[0] = b'S' as i32;
            g.decoder[1] = b'A' as i32;
            g.decoder[2] = b'B' as i32;
            g.decoder[3] = b'M' as i32;

            g.add_edge(s, a, 100, 0); // heavy branch into A
            g.add_edge(s, b, 1, 1); // light branch into B
            if reversed {
                g.add_edge(b, m, 5, 1);
                g.add_edge(a, m, 5, 0); // equal weight to B->M, added second
            } else {
                g.add_edge(a, m, 5, 0);
                g.add_edge(b, m, 5, 1); // equal weight to A->M, added second
            }

            g.topological_sort();
            let consensus = g.generate_consensus();

            // The heavier-scoring predecessor (A, via the 100-weight S->A edge) must win the
            // tie at M regardless of which equal-weight inedge (A->M or B->M) was added first;
            // B (the lighter branch) must NOT appear in the consensus.
            assert_eq!(
                consensus, "SAM",
                "reversed={reversed}: tie must resolve to the heavier predecessor (A), not \
                 insertion order"
            );
        }
    }

    #[test]
    fn coverage_counts_distinct_labels_across_in_and_out_edges() {
        // a -[label 0]-> b -[label 0,1]-> c
        let mut g = Graph::new();
        let a = g.add_node(0);
        let b = g.add_node(1);
        let c = g.add_node(2);

        g.add_edge(a, b, 1, 0);
        g.add_edge(b, c, 1, 0);
        g.add_edge(b, c, 1, 1);

        assert_eq!(g.nodes[a.0 as usize].coverage(&g), 1);
        assert_eq!(g.nodes[b.0 as usize].coverage(&g), 2);
        assert_eq!(g.nodes[c.0 as usize].coverage(&g), 2);
    }

    #[test]
    fn add_alignment_weight_with_empty_alignment_builds_linear_chain() {
        // First sequence into an empty graph always takes the empty-alignment fast path
        // (graph.cpp:175-179): a straight-line chain of one node per base.
        let mut g = Graph::new();
        g.add_alignment_weight(&[], b"ACGT", 1).unwrap();

        assert_eq!(g.nodes.len(), 4, "one node per base");
        assert_eq!(g.edges.len(), 3, "one edge between consecutive bases");
        assert_eq!(g.num_codes, 4, "four distinct symbols registered");
        assert_eq!(g.sequences.len(), 1, "one sequence recorded");
        assert_eq!(g.sequences[0], NodeId(0));

        // Interior edges sum the weights of both endpoints, matching AddSequence
        // (graph.cpp:104-106).
        for edge in &g.edges {
            assert_eq!(edge.weight, 2);
            assert_eq!(edge.labels, vec![0]);
        }
    }

    #[test]
    fn add_alignment_weight_aligns_mismatch_via_aligned_nodes_cross_link() {
        // Second sequence "AG" aligned against first sequence "AC": position 0 (A) matches an
        // existing node, position 1 (G) mismatches the existing C node and must fork into a new,
        // cross-linked aligned node (graph.cpp:207-231).
        let mut g = Graph::new();
        g.add_alignment_weight(&[], b"AC", 1).unwrap();
        let node_a = NodeId(0);
        let node_c = NodeId(1);

        g.add_alignment_weight(&[(0, 0), (1, 1)], b"AG", 1).unwrap();

        assert_eq!(g.nodes.len(), 3, "A, C, and a new G node");
        let node_g = NodeId(2);
        assert_eq!(
            g.nodes[node_g.0 as usize].code,
            g.coder[b'G' as usize] as u32
        );

        // Aligned-node cross-linking: C and G recorded as aligned to each other.
        assert_eq!(g.nodes[node_c.0 as usize].aligned_nodes, vec![node_g]);
        assert_eq!(g.nodes[node_g.0 as usize].aligned_nodes, vec![node_c]);

        // Two sequences recorded, both starting at node A.
        assert_eq!(g.sequences, vec![node_a, node_a]);

        // A->C edge (from the first sequence) is untouched; a new A->G edge carries the second
        // sequence's label and summed weight.
        let edge_ac = g
            .edges
            .iter()
            .find(|e| e.tail == node_a && e.head == node_c)
            .expect("A->C edge exists");
        assert_eq!(edge_ac.labels, vec![0]);
        assert_eq!(edge_ac.weight, 2);

        let edge_ag = g
            .edges
            .iter()
            .find(|e| e.tail == node_a && e.head == node_g)
            .expect("A->G edge exists");
        assert_eq!(edge_ag.labels, vec![1]);
        assert_eq!(edge_ag.weight, 2);
    }

    #[test]
    fn add_alignment_weight_threads_unaligned_prefix_and_suffix() {
        // Seed: "ACGT" -> nodes 0=A,1=C,2=G,3=T; codes A=0,C=1,G=2,T=3; sequences=[0].
        let mut g = Graph::new();
        g.add_alignment_weight(&[], b"ACGT", 1).unwrap();

        // Second sequence "TACGTA" maps ONLY its middle 'C' (seq index 2) to the existing C node
        // (node 1), forcing BOTH an unaligned prefix ("TA", indices 0-1) and an unaligned suffix
        // ("GTA", indices 3-5) through `add_sequence` (graph.cpp:196-243).
        g.add_alignment_weight(&[(1, 2)], b"TACGTA", 1).unwrap();

        // 4 seed + 2 prefix + 3 suffix = 9 nodes; the 'C' column reuses the existing node.
        assert_eq!(g.nodes.len(), 9);
        let node_c = NodeId(1);
        let (prefix_t, prefix_a) = (NodeId(4), NodeId(5));
        let (suffix_g, suffix_t, suffix_a) = (NodeId(6), NodeId(7), NodeId(8));

        // Returned start node is the prefix's FIRST node (graph.cpp:244, `begin`).
        assert_eq!(g.sequences, vec![NodeId(0), prefix_t]);

        // Node codes of the freshly created prefix/suffix nodes.
        assert_eq!(
            g.nodes[prefix_t.0 as usize].code,
            g.coder[b'T' as usize] as u32
        );
        assert_eq!(
            g.nodes[prefix_a.0 as usize].code,
            g.coder[b'A' as usize] as u32
        );
        assert_eq!(
            g.nodes[suffix_g.0 as usize].code,
            g.coder[b'G' as usize] as u32
        );
        assert_eq!(
            g.nodes[suffix_t.0 as usize].code,
            g.coder[b'T' as usize] as u32
        );
        assert_eq!(
            g.nodes[suffix_a.0 as usize].code,
            g.coder[b'A' as usize] as u32
        );

        // Every new edge belongs to sequence 1 with the summed adjacent-base weight (1+1=2).
        let find_edge = |tail: NodeId, head: NodeId| {
            g.edges
                .iter()
                .find(|e| e.tail == tail && e.head == head)
                .unwrap_or_else(|| panic!("edge {tail:?}->{head:?} exists"))
        };

        // Prefix chain, then the seam edge into the reused 'C' node.
        assert_eq!(find_edge(prefix_t, prefix_a).labels, vec![1]);
        assert_eq!(find_edge(prefix_t, prefix_a).weight, 2);
        assert_eq!(find_edge(prefix_a, node_c).labels, vec![1]);
        assert_eq!(find_edge(prefix_a, node_c).weight, 2);

        // Seam edge out of the reused 'C' node into the suffix chain, then the suffix chain.
        assert_eq!(find_edge(node_c, suffix_g).labels, vec![1]);
        assert_eq!(find_edge(node_c, suffix_g).weight, 2);
        assert_eq!(find_edge(suffix_g, suffix_t).labels, vec![1]);
        assert_eq!(find_edge(suffix_g, suffix_t).weight, 2);
        assert_eq!(find_edge(suffix_t, suffix_a).labels, vec![1]);
        assert_eq!(find_edge(suffix_t, suffix_a).weight, 2);

        // The reused 'C' node now sits on both sequences: original in/out plus the new seams.
        assert_eq!(g.nodes[node_c.0 as usize].inedges.len(), 2);
        assert_eq!(g.nodes[node_c.0 as usize].outedges.len(), 2);
    }

    #[test]
    fn add_alignment_rejects_unequal_weights() {
        let mut g = Graph::new();
        let err = g.add_alignment(&[], b"ACGT", &[1, 2, 3]).unwrap_err();
        assert!(matches!(err, GraphError::UnequalWeights));
    }

    #[test]
    fn add_alignment_rejects_alignment_missing_sequence_positions() {
        let mut g = Graph::new();
        g.add_alignment_weight(&[], b"AC", 1).unwrap();

        let err = g
            .add_alignment_weight(&[(0, -1), (1, -1)], b"AC", 1)
            .unwrap_err();
        assert!(matches!(err, GraphError::MissingSequence));
    }

    #[test]
    fn add_alignment_rejects_out_of_range_sequence_index() {
        let mut g = Graph::new();
        let err = g.add_alignment_weight(&[(-1, 5)], b"AC", 1).unwrap_err();
        assert!(matches!(err, GraphError::InvalidAlignment));
    }

    #[test]
    fn add_alignment_rejects_out_of_range_node_index() {
        // `node_idx` flows straight into `self.nodes[..]`; out-of-range values (too large for
        // the current node count, or below the `-1` "new node" sentinel) must return
        // InvalidAlignment rather than panic on the index.
        let mut g = Graph::new();
        assert!(matches!(
            g.add_alignment_weight(&[(5, 0)], b"AC", 1).unwrap_err(),
            GraphError::InvalidAlignment
        ));
        assert!(matches!(
            g.add_alignment_weight(&[(-2, 0)], b"AC", 1).unwrap_err(),
            GraphError::InvalidAlignment
        ));
    }

    #[test]
    fn initialize_msa_folds_aligned_peers_into_one_column() {
        // Seed "AC" (A->C), then align "AG" (forks C's mismatch into a new aligned node G,
        // graph.cpp:207-231), then align "AT" (extends the SAME aligned group by matching
        // against C's existing aligned_nodes, cross-linking a third node T to both C and G).
        // C, G, and T end up topo-adjacent in `rank_to_node` (aligned peers are emitted
        // immediately after their representative by `topological_sort`), which is exactly the
        // shape that exercises `InitializeMultipleSequenceAlignment`'s double-increment
        // (graph.cpp:325-331: the outer loop's `++i` PLUS the inner aligned_nodes loop's own
        // `++i` per peer) — a naive `enumerate()` port would instead give C, G, and T three
        // separate columns.
        let mut g = Graph::new();
        g.add_alignment_weight(&[], b"AC", 1).unwrap();
        g.add_alignment_weight(&[(0, 0), (1, 1)], b"AG", 1).unwrap();
        g.add_alignment_weight(&[(0, 0), (1, 1)], b"AT", 1).unwrap();

        assert_eq!(g.nodes.len(), 4, "sanity: A, C, G, T");
        let node_a = NodeId(0);
        let node_c = NodeId(1);
        let node_g = NodeId(2);
        let node_t = NodeId(3);
        assert_eq!(
            g.nodes[node_c.0 as usize].aligned_nodes.len(),
            2,
            "sanity: C's aligned group grew to include both G and T"
        );

        let (node_id_to_column, row_size) = g.initialize_msa();

        assert_eq!(
            row_size, 2,
            "row_size must count distinct columns (A's, and the aligned C/G/T group's), not \
             the 4 individual nodes"
        );
        assert_eq!(node_id_to_column[node_a.0 as usize], 0);

        let aligned_column = node_id_to_column[node_c.0 as usize];
        assert_ne!(aligned_column, node_id_to_column[node_a.0 as usize]);
        assert_eq!(
            node_id_to_column[node_g.0 as usize], aligned_column,
            "G must share C's column"
        );
        assert_eq!(
            node_id_to_column[node_t.0 as usize], aligned_column,
            "T must share C's column"
        );
    }

    /// Builds a small, deterministic (tie-free) graph: three sequences "AC", "AC", "AG" fed
    /// through `add_alignment_weight` in that order. The two "AC" sequences fold their C's
    /// into one shared node and double A->C's weight (4) over A->G's (2), so
    /// `traverse_heaviest_bundle`'s heaviest-bundle consensus is unambiguously "AC" (node G is
    /// never a tie candidate) — letting the GFA/DOT unit tests below assert an exact, fully
    /// predictable expected string without depending on any tie-break rule.
    fn small_gfa_dot_graph() -> Graph {
        let mut g = Graph::new();
        g.add_alignment_weight(&[], b"AC", 1).unwrap();
        g.add_alignment_weight(&[(0, 0), (1, 1)], b"AC", 1).unwrap();
        g.add_alignment_weight(&[(0, 0), (1, 1)], b"AG", 1).unwrap();
        g
    }

    #[test]
    fn to_gfa_emits_s_l_p_lines_with_one_based_ids_and_consensus_tags() {
        let mut g = small_gfa_dot_graph();
        assert_eq!(
            g.generate_consensus(),
            "AC",
            "sanity: deterministic consensus"
        );

        let headers = vec!["s0".to_string(), "s1".to_string(), "s2".to_string()];
        let gfa = g.to_gfa(&headers, &[], true);

        let expected = concat!(
            "H\tVN:Z:1.0\n",
            "S\t1\tA\tic:Z:true\n",
            "L\t1\t+\t2\t+\tOM\tew:f:4\tic:Z:true\n",
            "L\t1\t+\t3\t+\tOM\tew:f:2\n",
            "S\t2\tC\tic:Z:true\n",
            "S\t3\tG\n",
            "P\ts0\t1+,2+\t*\n",
            "P\ts1\t1+,2+\t*\n",
            "P\ts2\t1+,3+\t*\n",
            "P\tConsensus\t1+,2+\t*\n",
        );
        assert_eq!(gfa, expected);
    }

    #[test]
    fn to_gfa_reverses_path_and_flips_sign_when_is_reversed_is_set() {
        // Single sequence "ACG" -> nodes 1,2,3 (1-based). With `is_reversed[0] = true`, the P-line
        // path must be walked, then REVERSED, with every node's orientation suffix flipped from
        // `+` to `-` — mirroring `PrintGfa`'s `std::reverse(path)` + `(ir ? "-" : "+")`
        // (main.cpp:180-189). This exercises the reversal/sign-flip branch the corpus/`&[]` tests
        // never hit.
        let mut g = Graph::new();
        g.add_alignment_weight(&[], b"ACG", 1).unwrap();
        // No consensus generated: `consensus` stays empty, so no `ic:Z:true` tags appear and the
        // P-line reversal is the only thing under test.

        let headers = vec!["only".to_string()];
        let gfa = g.to_gfa(&headers, &[true], false);

        let expected = concat!(
            "H\tVN:Z:1.0\n",
            "S\t1\tA\n",
            "L\t1\t+\t2\t+\tOM\tew:f:2\n",
            "S\t2\tC\n",
            "L\t2\t+\t3\t+\tOM\tew:f:2\n",
            "S\t3\tG\n",
            "P\tonly\t3-,2-,1-\t*\n",
        );
        assert_eq!(gfa, expected);
    }

    #[test]
    fn to_dot_fills_consensus_nodes_colors_consensus_edges_and_dots_aligned_pairs() {
        let mut g = small_gfa_dot_graph();
        assert_eq!(
            g.generate_consensus(),
            "AC",
            "sanity: deterministic consensus"
        );

        let dot = g.to_dot();

        let expected = concat!(
            "digraph 3 {\n",
            "  graph [rankdir = LR]\n",
            "  0[label = \"0 - A\", style = filled, fillcolor = goldenrod1]\n",
            "  0 -> 1 [label = \"4\", color = goldenrod1]\n",
            "  0 -> 2 [label = \"2\"]\n",
            "  1[label = \"1 - C\", style = filled, fillcolor = goldenrod1]\n",
            "  1 -> 2 [style = dotted, arrowhead = none]\n",
            "  2[label = \"2 - G\"]\n",
            "}\n",
        );
        assert_eq!(dot, expected);
    }

    #[test]
    fn topological_sort_ranks_every_node_including_aligned_groups() {
        // Build a branching graph: seed "AC" (A->C), then align "AG" so its 'A' reuses the
        // existing A node but its 'G' mismatches the existing C node and forks into a new,
        // cross-linked aligned node (graph.cpp:207-231). This gives real in-edges (A->C, A->G)
        // plus a genuine aligned-node group ({C, G}), which is exactly the shape TopologicalSort
        // must handle: `add_alignment` already invokes `topological_sort` internally, so
        // `rank_to_node` should come out fully populated and valid without any direct call here.
        let mut g = Graph::new();
        g.add_alignment_weight(&[], b"AC", 1).unwrap();
        g.add_alignment_weight(&[(0, 0), (1, 1)], b"AG", 1).unwrap();

        assert_eq!(
            g.rank_to_node.len(),
            g.nodes.len(),
            "every node must be assigned a rank"
        );
        assert!(
            g.is_topologically_sorted(),
            "rank_to_node must be a valid topological order"
        );
    }

    /// A tiny hand-built graph: linear A->C->G->T (node ids 0..=3), no aligned peers.
    /// Extracting the window [1, 2] keeps nodes with id in {1, 2} reachable walking
    /// backwards from node `end`=2: node 2 (C? -> keep), its in-edge tail node 1, its
    /// aligned peers (none). Node 1's in-edge tail is node 0 (id 0 < begin=1 -> dropped).
    fn linear_graph_acgt() -> Graph {
        let mut g = Graph::new();
        let mut engine = crate::align::sisd::SisdEngine::new(
            crate::align::AlignmentType::Global,
            crate::align::Scoring::spoa_default(),
        );
        let seq = b"ACGT";
        let (aln, _s) = engine.align(seq, &g);
        g.add_alignment_weight(&aln, seq, 1).unwrap();
        g
    }

    #[test]
    fn subgraph_keeps_only_nodes_in_id_window_and_remaps() {
        let g = linear_graph_acgt();
        // Window [1, 2]: keep parent nodes 1 (C) and 2 (G).
        let (sub, sub_to_graph) = g.subgraph(NodeId(1), NodeId(2));
        assert_eq!(sub.num_nodes(), 2);
        // Map is indexed by subgraph node id; parent ids appear in ascending order
        // because Subgraph builds nodes by iterating the parent arena in id order.
        assert_eq!(sub_to_graph, vec![NodeId(1), NodeId(2)]);
        // Codes carried over: parent node 1 is 'C', node 2 is 'G'.
        assert_eq!(sub.decode(sub.node(NodeId(0)).code), Some(b'C'));
        assert_eq!(sub.decode(sub.node(NodeId(1)).code), Some(b'G'));
        // One internal edge (C->G); the edge into node 1 from node 0 is dropped
        // (node 0 is outside the window).
        assert_eq!(sub.num_edges(), 1);
        assert_eq!(sub.edge(EdgeId(0)).tail, NodeId(0));
        assert_eq!(sub.edge(EdgeId(0)).head, NodeId(1));
        // Faithful quirk: subgraph edges are labeled 0 (sequences not copied).
        assert_eq!(sub.edge(EdgeId(0)).labels, vec![0]);
        // num_codes / coder / decoder copied verbatim.
        assert_eq!(sub.num_codes(), g.num_codes());
        // Sequences are NOT copied.
        assert_eq!(sub.sequence_starts().len(), 0);
        // Topologically sorted.
        assert!(sub.is_topologically_sorted());
    }

    #[test]
    fn update_alignment_remaps_node_ids_and_passes_gaps() {
        let g = linear_graph_acgt();
        // Pretend a subgraph->graph map where subgraph ids 0,1 map to parent ids 2,3.
        let map = vec![NodeId(2), NodeId(3)];
        // Alignment pairs: (subgraph_node_id, seq_pos); -1 is a gap.
        let mut aln: crate::align::Alignment = vec![(-1, 0), (0, 1), (1, 2)];
        g.update_alignment(&map, &mut aln);
        assert_eq!(aln, vec![(-1, 0), (2, 1), (3, 2)]);
    }

    proptest! {
        /// Over random windows on a random small family: every kept subgraph node maps to a
        /// parent id `>= begin`, and every subgraph edge connects two kept nodes.
        ///
        /// Note there is no corresponding `<= end` upper bound to assert: `ExtractSubgraph`
        /// (`graph.cpp:551-561`) walks backwards from `end` over both in-edges *and* aligned-node
        /// cross-links, and aligned peers are not id-ordered (a mismatch column links a low-id
        /// node to the higher-id node created for the divergent base — see
        /// `topological_sort_ranks_every_node_including_aligned_groups` above for exactly this
        /// shape). So a low-id `end` can pull in a higher-id aligned peer, whose own in-edges are
        /// then walked too; the C++ oracle has no upper-bound check (only `curr->id >= end->id`,
        /// the window's *lower* bound, gates inclusion), and this port is faithful to that.
        #[test]
        fn subgraph_kept_ids_in_window_and_edges_internal(
            seqs in proptest::collection::vec("[ACGT]{1,12}", 2..6),
            a in 0u32..40,
            b in 0u32..40,
        ) {
            let mut g = Graph::new();
            let mut engine = crate::align::sisd::SisdEngine::new(
                crate::align::AlignmentType::Global,
                crate::align::Scoring::spoa_default(),
            );
            for s in &seqs {
                let bytes = s.as_bytes();
                let (aln, _) = engine.align(bytes, &g);
                g.add_alignment_weight(&aln, bytes, 1).unwrap();
            }
            prop_assume!(g.num_nodes() > 0);
            let n = g.num_nodes() as u32;
            let begin = NodeId(a % n);
            let end = NodeId(b % n);

            let (sub, map) = g.subgraph(begin, end);
            // Map length equals subgraph node count.
            prop_assert_eq!(map.len(), sub.num_nodes());
            // Every kept parent id respects the window's lower bound.
            for &parent_id in &map {
                prop_assert!(parent_id.0 >= begin.0);
            }
            // Every subgraph edge connects two kept nodes (ids < subgraph node count).
            for e in sub.edges() {
                prop_assert!((e.tail.0 as usize) < sub.num_nodes());
                prop_assert!((e.head.0 as usize) < sub.num_nodes());
            }
            prop_assert!(sub.is_topologically_sorted());
        }
    }
}
