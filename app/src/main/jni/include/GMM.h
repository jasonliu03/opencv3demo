#include "Color.h"
#include "GaussianFitter.h"
#include <vector>
using namespace std;
typedef unsigned int uint;
class GMM {
 public:
  GMM(unsigned int K);
  ~GMM();
  unsigned int K() const { return mK; }
  Real p(Color c);
  Real p(unsigned int i, Color c);
  int Build(double** data, uint nrows);
 private:
  unsigned int mK;
  GaussianPDF* mGaussians;
};
