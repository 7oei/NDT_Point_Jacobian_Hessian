#include <bits/stdc++.h>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <Eigen/Core>
#include <Eigen/LU>
#include <boost/python/numpy/dtype.hpp>
#include <boost/python/numpy/ndarray.hpp>
#include <boost/python/tuple.hpp>
#include "PointDerivatives.hpp"

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
    mu << 1, 0, 0;
    cov = MatrixXd::Identity(3, 3);

  }

  p::object calcScore(np::ndarray &point_py, np::ndarray &transform_py){
    // convert ndarray to EigenMatrix.
    // MatrixXd point = convertNdarrayToEigen(point_py);
    PointT point = convertNdarrayToEigen(point_py);
    // MatrixXd transform = convertNdarrayToEigen(transform_py);
    TransformT transform = convertNdarrayToEigen(transform_py);
    cout << "point" << endl;
    cout << point << endl;
    cout << "transform" << endl;
    cout << transform << endl;
    // TODO: calc score and jacobian and hessian!
    Matrix<double, 3, 3> cov_inv = cov.inverse();
    cout << "cov_inv" << endl;
    cout << cov_inv << endl;
    cout << "mu" << endl;
    cout << mu << endl;
    
    double output_prob = 0.055;
    double resolution = 1.0;
    Matrix<double, 3, 1> x_k_dash = point - mu;

    JacobianCoefficientsT jacobian_coefficients;
    HessianCoefficientsT hessian_coefficients;
    angleDerivatives(transform,jacobian_coefficients,hessian_coefficients);

    PointJacobianT point_jacobian;
    PointHessianT point_hessian;
    pointDerivatives(point,jacobian_coefficients,hessian_coefficients,point_jacobian,point_hessian);

    ScoreParams params(output_prob, resolution);

    tuple<double, JacobianT, HessianT> derivatives = computeDerivative(params, point_jacobian, point_hessian, x_k_dash, cov_inv);


    // MatrixXd jacobian = MatrixXd::Zero(3, 6);
    // MatrixXd hessian = MatrixXd::Zero(18, 6);    
    MatrixXd jacobian = MatrixXd::Zero(6, 1);
    MatrixXd hessian = MatrixXd::Zero(6, 6);
    jacobian = get<1>(derivatives);
    hessian = get<2>(derivatives);

    cout << "jacobian" << endl;
    cout << jacobian << endl;
    cout << "hessian" << endl;
    cout << hessian << endl;

    float score = get<0>(derivatives);
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
