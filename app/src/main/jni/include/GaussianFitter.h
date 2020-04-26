#include "Color.h"
#include "Global.h"
struct GaussianPDF {
  Color mu;
  Real covariance[3][3];
  Real determinant;
  Real inverse[3][3];
  Real pi;
};
class GaussianFitter {
 public:
  GaussianFitter();
  void add(Color c);
  void finalize(GaussianPDF& g, unsigned int totalCount) const;

 private:
  Color s;
  Real p[3][3];
  unsigned int count;
};
