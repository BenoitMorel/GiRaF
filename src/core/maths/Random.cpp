#include "Random.hpp"

std::mt19937_64 Random::_rng;
std::uniform_int_distribution<int> Random::_unii(0);
std::uniform_real_distribution<double> Random::_uniproba;

void Random::setSeed(unsigned int seed)
{
  _rng.seed(seed);
}
int Random::getInt() 
{
  return _unii(_rng);
}

int Random::getInt(int max)
{
  return getInt() % max;
}

double Random::getProba() 
{
  return _uniproba(_rng);
}


