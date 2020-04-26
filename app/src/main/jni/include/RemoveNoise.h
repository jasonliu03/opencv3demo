#include "Global.h"
class RemoveNoise {
 public:
  RemoveNoise();
  ~RemoveNoise();
  void LessConnectedRegionRemove(IplImage* image, int area);
  int RemoveCrackImageNoise(IplImage* image, int area, int lineLength);
 private:
};
