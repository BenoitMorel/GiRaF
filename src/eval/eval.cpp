#include <iostream>
#include <corax/corax.h>
#include <IO/Logger.hpp>
#include <trees/PLLUnrootedTree.hpp>
#include <parallelization/ParallelContext.hpp>
#include <maths/bitvector.hpp>
#include <unordered_set>
#include <util/types.hpp>

using SplitTable = std::unordered_set<BitVector>;


StringToUint getSpeciesToSpid(PLLUnrootedTree &tree) 
{
  StringToUint res;
  for (auto leaf: tree.getLeaves()) {
    res.insert({std::string(leaf->label), res.size()});
  }
  return res;
}

SplitTable getSplitTable(const PLLUnrootedTree &tree,
    const StringToUint &speciesToSpid)
{
  std::vector<BitVector> splits(tree.getDirectedNodesNumber());
  SplitTable table;
  for (auto node: tree.getPostOrderNodes()) {
    auto id = node->node_index;
    splits[id] = BitVector(tree.getLeavesNumber(), false);
    if (!node->next) {
      auto label = std::string(node->label);
      auto spid = speciesToSpid.at(label);
      splits[id].set(spid); 
    } else {
      auto idLeft = PLLUnrootedTree::getLeft(node)->node_index;
      auto idRight = PLLUnrootedTree::getRight(node)->node_index;
      splits[id] = (splits[idLeft] | splits[idRight]);
      if (node->back->next) { // trivial partition
        if (splits[id].get(0)) {
          table.insert(splits[id]);
        } else {
          table.insert(~splits[id]);
        }
      }
    }
  }
  return table;
}

double computeRFD(const PLLUnrootedTree &tree1,
    const PLLUnrootedTree &tree2,
    const StringToUint &speciesToSpid)
{
  auto splits1 = getSplitTable(tree1, speciesToSpid);
  auto splits2 = getSplitTable(tree2, speciesToSpid);
  double distance = 0.0;
  for (auto sp: splits1) {
    if (splits2.find(sp) == splits2.end()) {
      distance += 1.0;
    }
  }
  for (auto sp: splits2) {
    if (splits1.find(sp) == splits1.end()) {
      distance += 1.0;
    } 
  }
  return distance; 
}

double computeNormalizedRFD(const PLLUnrootedTree &tree1,
    const PLLUnrootedTree &tree2,
    const StringToUint & speciesToSpid)
{
  auto distance = computeRFD(tree1, tree2, speciesToSpid);
  auto branches1 = tree1.getLeavesNumber() - 3;
  auto branches2 = tree2.getLeavesNumber() - 3;
  return distance / (branches1 + branches2);
}

double splitDistance(const BitVector &v1, 
    const BitVector &v2) 
{
  auto xor1 = v1 ^ v2;
  auto xor2 = v1 ^ (~v2);
  auto d = static_cast<double>(std::min(xor1.count(), xor2.count())) * 2.0;
  return d;
}

double auxGRFNaive(const SplitTable &splits1,
    const SplitTable &splits2)
{
  double distance = 0.0;
  size_t unionSize = splits1.size();
  
  for (const auto &sp2: splits2) {
    if (splits1.find(sp2) == splits1.end()) {
      unionSize += 1;
      for (const auto &sp1: splits1) {
        distance += splitDistance(sp1, sp2);
      }
    }
  }
  double weight = static_cast<double>(unionSize * splits1.size());
  return distance / weight;
}

double computeGRFDNaive(const PLLUnrootedTree &tree1, 
    const PLLUnrootedTree &tree2, 
    const StringToUint &speciesToSpid)
{
  auto splits1 = getSplitTable(tree1, speciesToSpid);
  auto splits2 = getSplitTable(tree2, speciesToSpid);
  double distance = 0.0;
  distance += auxGRFNaive(splits1, splits2);
  distance += auxGRFNaive(splits2, splits1);
  return distance;
}

int main(int argc, char * argv[])
{
  Logger::init();
  if (argc != 3) {
    Logger::error << "Syntax: tree1 tree2" << std::endl;
    ParallelContext::abort(1);
  }
  std::string treePath1(argv[1]);
  std::string treePath2(argv[2]);
  PLLUnrootedTree tree1(treePath1);
  PLLUnrootedTree tree2(treePath2);
  auto speciesToSpid = getSpeciesToSpid(tree1);
  Logger::info << 
    computeNormalizedRFD(tree1, tree2, speciesToSpid) << 
    std::endl;
  Logger::info <<  
    computeGRFDNaive(tree1, tree2, speciesToSpid) << 
    std::endl;
  return 0;

}
