#include <bits/stdc++.h>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <Eigen/Core>

namespace np = boost::python::numpy;
namespace p = boost::python;

using namespace std;


vector<Eigen>


class NDT{
public:
  void create_map(np::ndarray &reference_pc_py){
    // TODO: convert list of EigenMatrix.

  }

  void registration(np::ndarray &scan_pc_py){
    // TODO: convert list of EigenMatrix
  }
};


BOOST_PYTHON_MODULE(libndt_sample)
{
  Py_Initialize();
  np::initialize();

  p::class_<NDT>("NDT")
    .def("create_map", &NDT::create_map);
}
