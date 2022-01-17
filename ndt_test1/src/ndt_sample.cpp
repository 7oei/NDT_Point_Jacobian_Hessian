#include <bits/stdc++.h>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <Eigen/Core>
#include <boost/python/numpy/dtype.hpp>
#include <boost/python/numpy/ndarray.hpp>
#include <boost/python/tuple.hpp>

namespace np = boost::python::numpy;
namespace p = boost::python;

using namespace std;
using namespace Eigen;


MatrixXd convertNdarrayToEigen(const np::ndarray &n){
  int length = n.shape(0);

  float *p = reinterpret_cast<float *>(n.get_data());

  MatrixXd result = MatrixXd::Zero(length, 1);

  for(int i = 0; i < length; ++i){
    result(i, 0) = p[i];
  }

  return result;
}

p::list convertEigenToList(const MatrixXd &src){

  unsigned int r = src.rows();
  unsigned int c = src.cols();

  p::list dest;

  for(unsigned i = 0; i < c; ++i){
    for(unsigned j = 0; j < r; ++j){
      dest.append(src(j, i));
    }
  }

  return dest;
}

class NDT{
public:
  NDT(){
    setMap();
  }

  void setMap(){
    mu << 0, 0, 0;
    cov = MatrixXd::Identity(3, 3);

  }

  p::object calcScore(np::ndarray &point_py, np::ndarray &transform_py){
    // convert ndarray to EigenMatrix.
    MatrixXd point = convertNdarrayToEigen(point_py);
    MatrixXd transform = convertNdarrayToEigen(transform_py);

    cout << point << endl;
    cout << transform << endl;

    // TODO: calc score and jacobian and hessian!

    MatrixXd jacobian = MatrixXd::Zero(3, 6);
    MatrixXd hessian = MatrixXd::Zero(18, 6);

    jacobian(2, 3) = 1.0;
    hessian(2, 3) = 1.0;

    cout << "jacobian" << endl;
    cout << jacobian << endl;

    cout << "hessian" << endl;
    cout << hessian << endl;

    float score = 0;
    p::list jacobian_py = convertEigenToList(jacobian);
    p::list hessian_py = convertEigenToList(hessian);

    return p::make_tuple(score, jacobian_py, hessian_py);
  }

private:
  Matrix<double, 3, 1> mu;
  Matrix<double, 3, 3> cov;
};


BOOST_PYTHON_MODULE(libndt_sample)
{
  Py_Initialize();
  np::initialize();

  p::class_<NDT>("NDT")
    .def("calcScore", &NDT::calcScore);
}
