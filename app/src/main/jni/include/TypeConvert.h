#include "Global.h"
template <class T>
string toString(const T& value) {
  stringstream ss;
  string str;
  ss << value;
  ss >> str;
  return str;
}
template <class T>
T fromString(const string& str) {
  stringstream ss;
  T value;
  ss << str;
  ss >> value;
  return value;
}
