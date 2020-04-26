#ifndef COLOR_H
#define COLOR_H
#include "Global.h"
#define Real float
class Color {
 public:
  Color() : r(0), g(0), b(0) {}
  Color(Real _r, Real _g, Real _b) : r(_r), g(_g), b(_b) {}
  Real r, g, b;
};
Real Distance(unsigned int x1, unsigned int y1, unsigned int x2,
              unsigned int y2);
Real ColorDistance2(const Color& c1, const Color& c2);
#endif
