// spoa_oracle: a dev/CI-only differential-testing helper.
//
// Links the pinned upstream spoa submodule (forced to its SISD/scalar
// alignment engine — see oracle/CMakeLists.txt) and, for each JSONL case read
// on stdin, drives the exact same sequence of spoa::AlignmentEngine /
// spoa::Graph calls that third_party/spoa/src/main.cpp performs, emitting one
// JSONL result line on stdout per case. This lets the Rust reimplementation's
// test suite assert bit-exact parity against upstream spoa without linking
// spoa into the Rust crate itself.
//
// Input (one JSON object per line):
//   {"id": 0, "type": "NW", "m": 5, "n": -4, "g": -8, "e": -6, "q": -10,
//    "c": -4, "seqs": ["ACGT", "AGT"], "quals": null, "min_coverage": -1}
//
// Output (one JSON object per line, order preserved, keyed by "id"):
//   {"id": 0,
//    "alignments": [[[-1, 0], [0, 1]]],
//    "consensus": "ACGT",
//    "msa": ["ACGT", "A-GT"],
//    "gfa": "H\tVN:Z:1.0\n...",
//    "dot": "digraph 2 {\n..."}
//
// This file has no dependency beyond the C++ standard library and the spoa
// library target: JSON is parsed and emitted by hand (see JsonValue /
// JsonParser / JsonEscape below).

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "spoa/spoa.hpp"

namespace {

// ---------------------------------------------------------------------------
// Minimal hand-rolled JSON reader, sufficient for the fixed oracle request
// schema (flat object of numbers/strings/null/arrays of strings). Kept
// dependency-free per the task brief rather than vendoring a JSON library.
// ---------------------------------------------------------------------------

struct JsonValue {
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };

  Type type = Type::kNull;
  bool boolean = false;
  double number = 0.0;
  std::string str;
  std::vector<JsonValue> array;
  std::vector<std::pair<std::string, JsonValue>> object;

  bool IsNull() const { return type == Type::kNull; }

  bool Has(const std::string& key) const {
    for (const auto& kv : object) {
      if (kv.first == key) {
        return true;
      }
    }
    return false;
  }

  const JsonValue& At(const std::string& key) const {
    for (const auto& kv : object) {
      if (kv.first == key) {
        return kv.second;
      }
    }
    throw std::runtime_error("[spoa_oracle] missing JSON field: " + key);
  }

  std::int64_t AsInt() const {
    if (type != Type::kNumber) {
      throw std::runtime_error("[spoa_oracle] expected a JSON number");
    }
    return static_cast<std::int64_t>(number);
  }

  const std::string& AsString() const {
    if (type != Type::kString) {
      throw std::runtime_error("[spoa_oracle] expected a JSON string");
    }
    return str;
  }
};

class JsonParser {
 public:
  explicit JsonParser(const std::string& text) : text_(text), pos_(0) {}

  JsonValue Parse() {
    JsonValue value = ParseValue();
    SkipWhitespace();
    return value;
  }

 private:
  const std::string& text_;
  std::size_t pos_;

  char Peek() const { return pos_ < text_.size() ? text_[pos_] : '\0'; }

  char Advance() { return text_[pos_++]; }

  void SkipWhitespace() {
    while (pos_ < text_.size() &&
           std::isspace(static_cast<unsigned char>(text_[pos_]))) {
      ++pos_;
    }
  }

  void Expect(char c) {
    if (Peek() != c) {
      std::ostringstream oss;
      oss << "[spoa_oracle] JSON parse error: expected '" << c
          << "' at offset " << pos_;
      throw std::runtime_error(oss.str());
    }
    ++pos_;
  }

  bool Consume(const std::string& literal) {
    if (text_.compare(pos_, literal.size(), literal) == 0) {
      pos_ += literal.size();
      return true;
    }
    return false;
  }

  JsonValue ParseValue() {
    SkipWhitespace();
    switch (Peek()) {
      case '{':
        return ParseObject();
      case '[':
        return ParseArray();
      case '"':
        return ParseString();
      case 't':
      case 'f':
        return ParseBool();
      case 'n':
        return ParseNull();
      default:
        return ParseNumber();
    }
  }

  JsonValue ParseObject() {
    JsonValue value;
    value.type = JsonValue::Type::kObject;
    Expect('{');
    SkipWhitespace();
    if (Peek() == '}') {
      ++pos_;
      return value;
    }
    while (true) {
      SkipWhitespace();
      JsonValue key = ParseString();
      SkipWhitespace();
      Expect(':');
      JsonValue val = ParseValue();
      value.object.emplace_back(key.str, std::move(val));
      SkipWhitespace();
      if (Peek() == ',') {
        ++pos_;
        continue;
      }
      break;
    }
    SkipWhitespace();
    Expect('}');
    return value;
  }

  JsonValue ParseArray() {
    JsonValue value;
    value.type = JsonValue::Type::kArray;
    Expect('[');
    SkipWhitespace();
    if (Peek() == ']') {
      ++pos_;
      return value;
    }
    while (true) {
      value.array.emplace_back(ParseValue());
      SkipWhitespace();
      if (Peek() == ',') {
        ++pos_;
        continue;
      }
      break;
    }
    SkipWhitespace();
    Expect(']');
    return value;
  }

  JsonValue ParseString() {
    JsonValue value;
    value.type = JsonValue::Type::kString;
    Expect('"');
    std::string out;
    while (true) {
      if (pos_ >= text_.size()) {
        throw std::runtime_error("[spoa_oracle] JSON parse error: unterminated string");  // NOLINT
      }
      char c = Advance();
      if (c == '"') {
        break;
      }
      if (c == '\\') {
        if (pos_ >= text_.size()) {
          throw std::runtime_error("[spoa_oracle] JSON parse error: bad escape");  // NOLINT
        }
        char esc = Advance();
        switch (esc) {
          case '"': out += '"'; break;
          case '\\': out += '\\'; break;
          case '/': out += '/'; break;
          case 'b': out += '\b'; break;
          case 'f': out += '\f'; break;
          case 'n': out += '\n'; break;
          case 'r': out += '\r'; break;
          case 't': out += '\t'; break;
          case 'u': {
            if (pos_ + 4 > text_.size()) {
              throw std::runtime_error("[spoa_oracle] JSON parse error: bad \\u escape");  // NOLINT
            }
            unsigned int code = 0;
            for (int i = 0; i < 4; ++i) {
              code <<= 4;
              char h = Advance();
              if (h >= '0' && h <= '9') {
                code |= static_cast<unsigned int>(h - '0');
              } else if (h >= 'a' && h <= 'f') {
                code |= static_cast<unsigned int>(h - 'a' + 10);
              } else if (h >= 'A' && h <= 'F') {
                code |= static_cast<unsigned int>(h - 'A' + 10);
              } else {
                throw std::runtime_error("[spoa_oracle] JSON parse error: bad hex digit");  // NOLINT
              }
            }
            // Sufficient for this oracle's schema (ASCII DNA/quality data);
            // encode the BMP code point as UTF-8 without surrogate-pair
            // handling.
            if (code < 0x80) {
              out += static_cast<char>(code);
            } else if (code < 0x800) {
              out += static_cast<char>(0xC0 | (code >> 6));
              out += static_cast<char>(0x80 | (code & 0x3F));
            } else {
              out += static_cast<char>(0xE0 | (code >> 12));
              out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
              out += static_cast<char>(0x80 | (code & 0x3F));
            }
            break;
          }
          default:
            throw std::runtime_error("[spoa_oracle] JSON parse error: unknown escape");  // NOLINT
        }
      } else {
        out += c;
      }
    }
    value.str = std::move(out);
    return value;
  }

  JsonValue ParseBool() {
    JsonValue value;
    value.type = JsonValue::Type::kBool;
    if (Consume("true")) {
      value.boolean = true;
    } else if (Consume("false")) {
      value.boolean = false;
    } else {
      throw std::runtime_error("[spoa_oracle] JSON parse error: bad literal");
    }
    return value;
  }

  JsonValue ParseNull() {
    if (!Consume("null")) {
      throw std::runtime_error("[spoa_oracle] JSON parse error: bad literal");
    }
    return JsonValue{};
  }

  JsonValue ParseNumber() {
    std::size_t start = pos_;
    if (Peek() == '-') {
      ++pos_;
    }
    while (std::isdigit(static_cast<unsigned char>(Peek()))) {
      ++pos_;
    }
    if (Peek() == '.') {
      ++pos_;
      while (std::isdigit(static_cast<unsigned char>(Peek()))) {
        ++pos_;
      }
    }
    if (Peek() == 'e' || Peek() == 'E') {
      ++pos_;
      if (Peek() == '+' || Peek() == '-') {
        ++pos_;
      }
      while (std::isdigit(static_cast<unsigned char>(Peek()))) {
        ++pos_;
      }
    }
    if (pos_ == start) {
      throw std::runtime_error("[spoa_oracle] JSON parse error: bad number");
    }
    JsonValue value;
    value.type = JsonValue::Type::kNumber;
    value.number = std::stod(text_.substr(start, pos_ - start));
    return value;
  }
};

// Escapes a string for embedding as a JSON string value: quotes, backslashes,
// and control characters (newlines/tabs/CRs from GFA, and the literal `"`
// characters DOT node labels contain — see graph.cpp:655-656) all need
// escaping or the downstream Rust serde_json parse will fail/mis-parse.
std::string JsonEscape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\t': out += "\\t"; break;
      case '\r': out += "\\r"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Oracle request/response handling.
// ---------------------------------------------------------------------------

spoa::AlignmentType ParseAlignmentType(const std::string& type) {
  if (type == "SW") {
    return spoa::AlignmentType::kSW;
  }
  if (type == "NW") {
    return spoa::AlignmentType::kNW;
  }
  if (type == "OV") {
    return spoa::AlignmentType::kOV;
  }
  throw std::runtime_error("[spoa_oracle] unknown alignment type: " + type);
}

// Reproduces spoa::Graph::PrintDot (third_party/spoa/src/graph.cpp:640-680)
// into a string instead of a file.
std::string PrintDotToString(const spoa::Graph& graph) {
  std::ostringstream os;

  std::vector<std::int32_t> consensus_rank(graph.nodes().size(), -1);
  std::int32_t rank = 0;
  for (const auto& it : graph.consensus()) {
    consensus_rank[it->id] = rank++;
  }

  os << "digraph " << graph.sequences().size() << " {\n"
     << "  graph [rankdir = LR]\n";
  for (const auto& it : graph.nodes()) {
    os << "  " << it->id << "[label = \"" << it->id << " - "
       << static_cast<char>(graph.decoder(it->code)) << "\"";
    if (consensus_rank[it->id] != -1) {
      os << ", style = filled, fillcolor = goldenrod1";
    }
    os << "]\n";

    for (const auto& jt : it->outedges) {
      os << "  " << it->id << " -> " << jt->head->id << " [label = \""
         << jt->weight << "\"";
      if (consensus_rank[it->id] + 1 == consensus_rank[jt->head->id]) {
        os << ", color = goldenrod1";
      }
      os << "]\n";
    }
    for (const auto& jt : it->aligned_nodes) {
      if (jt->id > it->id) {
        os << "  " << it->id << " -> " << jt->id
           << " [style = dotted, arrowhead = none]\n";
      }
    }
  }
  os << "}\n";

  return os.str();
}

// Reproduces main.cpp's PrintGfa (third_party/spoa/src/main.cpp:123-203) into
// a string instead of stdout. Always called with include_consensus = false
// and an empty is_reversed vector, matching how the Rust parity tests call
// Graph::to_gfa (docs/superpowers/plans/2026-07-02-spoars-scalar-library-and-cli.md).  // NOLINT
//
// `headers` are the real per-sequence names when the request's optional
// "names" field is present (see ParseCase / RunCase below), or a 0-based
// sequence-index decimal-string placeholder otherwise.
std::string PrintGfaToString(const spoa::Graph& graph,
                              const std::vector<std::string>& headers) {
  std::ostringstream os;

  std::vector<bool> is_consensus_node(graph.nodes().size(), false);
  for (const auto& it : graph.consensus()) {
    is_consensus_node[it->id] = true;
  }

  os << "H\tVN:Z:1.0\n";
  for (const auto& it : graph.nodes()) {
    os << "S\t" << it->id + 1 << "\t"
       << static_cast<char>(graph.decoder(it->code));
    if (is_consensus_node[it->id]) {
      os << "\tic:Z:true";
    }
    os << "\n";

    for (const auto& jt : it->outedges) {
      os << "L\t" << it->id + 1 << "\t"
         << "+\t" << jt->head->id + 1 << "\t"
         << "+\t"
         << "OM\t"
         << "ew:f:" << jt->weight;
      if (is_consensus_node[it->id] && is_consensus_node[jt->head->id]) {
        os << "\tic:Z:true";
      }
      os << "\n";
    }
  }

  for (std::uint32_t i = 0; i < graph.sequences().size(); ++i) {
    os << "P\t" << headers[i] << "\t";

    std::vector<std::uint32_t> path;
    auto curr = graph.sequences()[i];
    while (true) {
      path.emplace_back(curr->id + 1);
      curr = curr->Successor(i);
      if (!curr) {
        break;
      }
    }

    for (std::uint32_t j = 0; j < path.size(); ++j) {
      if (j != 0) {
        os << ",";
      }
      os << path[j] << "+";
    }
    os << "\t*\n";
  }

  return os.str();
}

// One parsed request line from the Oracle JSON contract.
struct OracleCase {
  std::int64_t id = 0;
  spoa::AlignmentType type = spoa::AlignmentType::kNW;
  std::int8_t m = 0;
  std::int8_t n = 0;
  std::int8_t g = 0;
  std::int8_t e = 0;
  std::int8_t q = 0;
  std::int8_t c = 0;
  std::vector<std::string> seqs;
  bool has_quals = false;
  std::vector<std::string> quals;
  std::int32_t min_coverage = -1;
  bool has_names = false;
  std::vector<std::string> names;
  bool has_subgraph = false;
  std::uint32_t subgraph_begin = 0;
  std::uint32_t subgraph_end = 0;
};

OracleCase ParseCase(const JsonValue& v) {
  OracleCase result;
  result.id = v.At("id").AsInt();
  result.type = ParseAlignmentType(v.At("type").AsString());
  result.m = static_cast<std::int8_t>(v.At("m").AsInt());
  result.n = static_cast<std::int8_t>(v.At("n").AsInt());
  result.g = static_cast<std::int8_t>(v.At("g").AsInt());
  result.e = static_cast<std::int8_t>(v.At("e").AsInt());
  result.q = static_cast<std::int8_t>(v.At("q").AsInt());
  result.c = static_cast<std::int8_t>(v.At("c").AsInt());

  const JsonValue& seqs = v.At("seqs");
  for (const auto& s : seqs.array) {
    result.seqs.emplace_back(s.AsString());
  }

  const JsonValue& quals = v.At("quals");
  if (!quals.IsNull()) {
    result.has_quals = true;
    for (const auto& q : quals.array) {
      result.quals.emplace_back(q.AsString());
    }
    if (result.quals.size() != result.seqs.size()) {
      throw std::runtime_error(
          "[spoa_oracle] quals length does not match seqs length");
    }
  }

  result.min_coverage = static_cast<std::int32_t>(v.At("min_coverage").AsInt());  // NOLINT

  if (v.Has("names") && !v.At("names").IsNull()) {
    result.has_names = true;
    for (const auto& n : v.At("names").array) {
      result.names.emplace_back(n.AsString());
    }
    if (result.names.size() != result.seqs.size()) {
      throw std::runtime_error(
          "[spoa_oracle] names length does not match seqs length");
    }
  }

  if (v.Has("subgraph") && !v.At("subgraph").IsNull()) {
    const JsonValue& sg = v.At("subgraph");
    if (sg.array.size() != 2) {
      throw std::runtime_error(
          "[spoa_oracle] subgraph must be a 2-element [begin, end] array");
    }
    result.has_subgraph = true;
    result.subgraph_begin = static_cast<std::uint32_t>(sg.array[0].AsInt());
    result.subgraph_end = static_cast<std::uint32_t>(sg.array[1].AsInt());
  }

  return result;
}

// Runs one oracle case through spoa and writes the JSONL result line.
void RunCase(const OracleCase& oc, std::ostream& out) {
  auto engine = spoa::AlignmentEngine::Create(
      oc.type, oc.m, oc.n, oc.g, oc.e, oc.q, oc.c);

  spoa::Graph graph{};

  std::vector<spoa::Alignment> alignments;
  alignments.reserve(oc.seqs.size());

  for (std::size_t i = 0; i < oc.seqs.size(); ++i) {
    spoa::Alignment alignment = engine->Align(oc.seqs[i], graph);
    alignments.emplace_back(alignment);

    if (oc.has_quals) {
      graph.AddAlignment(alignment, oc.seqs[i], oc.quals[i]);
    } else {
      graph.AddAlignment(alignment, oc.seqs[i]);
    }
  }

  std::string consensus = graph.GenerateConsensus(oc.min_coverage);
  std::vector<std::string> msa =
      graph.GenerateMultipleSequenceAlignment(false);

  std::vector<std::string> headers;
  headers.reserve(oc.seqs.size());
  if (oc.has_names) {
    headers = oc.names;
  } else {
    for (std::size_t i = 0; i < oc.seqs.size(); ++i) {
      headers.emplace_back(std::to_string(i));
    }
  }
  std::string gfa = PrintGfaToString(graph, headers);
  std::string dot = PrintDotToString(graph);

  std::string subgraph_json;
  if (oc.has_subgraph) {
    std::vector<const spoa::Graph::Node*> map;
    spoa::Graph subgraph = graph.Subgraph(oc.subgraph_begin, oc.subgraph_end, &map);

    std::ostringstream sg;
    sg << ",\"subgraph\":{";
    // map: parent graph id per subgraph node id (only the first num_nodes entries are set).
    sg << "\"map\":[";
    for (std::size_t i = 0; i < subgraph.nodes().size(); ++i) {
      if (i != 0) sg << ",";
      sg << map[i]->id;
    }
    sg << "],\"codes\":[";
    for (std::size_t i = 0; i < subgraph.nodes().size(); ++i) {
      if (i != 0) sg << ",";
      sg << subgraph.nodes()[i]->code;
    }
    sg << "],\"edges\":[";
    for (std::size_t i = 0; i < subgraph.edges().size(); ++i) {
      if (i != 0) sg << ",";
      const auto& e = subgraph.edges()[i];
      sg << "[" << e->tail->id << "," << e->head->id << "," << e->weight << "]";
    }
    sg << "],\"rank\":[";
    for (std::size_t i = 0; i < subgraph.rank_to_node().size(); ++i) {
      if (i != 0) sg << ",";
      sg << subgraph.rank_to_node()[i]->id;
    }
    sg << "],\"aligned\":[";
    bool first_aligned = true;
    for (const auto& n : subgraph.nodes()) {
      for (const auto& a : n->aligned_nodes) {
        if (!first_aligned) sg << ",";
        first_aligned = false;
        sg << "[" << n->id << "," << a->id << "]";
      }
    }
    sg << "]}";
    subgraph_json = sg.str();
  }

  out << "{\"id\":" << oc.id << ",\"alignments\":[";
  for (std::size_t i = 0; i < alignments.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << "[";
    const spoa::Alignment& alignment = alignments[i];
    for (std::size_t j = 0; j < alignment.size(); ++j) {
      if (j != 0) {
        out << ",";
      }
      out << "[" << alignment[j].first << "," << alignment[j].second << "]";
    }
    out << "]";
  }
  out << "],\"consensus\":\"" << JsonEscape(consensus) << "\",\"msa\":[";
  for (std::size_t i = 0; i < msa.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << "\"" << JsonEscape(msa[i]) << "\"";
  }
  out << "],\"gfa\":\"" << JsonEscape(gfa) << "\",\"dot\":\""
      << JsonEscape(dot) << "\"" << subgraph_json << "}\n";
  out.flush();
}

}  // namespace

int main() {
  std::string line;
  while (std::getline(std::cin, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
      continue;  // skip blank lines
    }

    try {
      JsonParser parser(line);
      JsonValue request = parser.Parse();
      OracleCase oc = ParseCase(request);
      RunCase(oc, std::cout);
    } catch (const std::exception& ex) {
      std::cerr << "[spoa_oracle] error: " << ex.what() << std::endl;
      return 1;
    }
  }

  return 0;
}
