#ifndef POINT_DERIVATIVES_H
#define POINT_DERIVATIVES_H

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
#include <math.h>
#include <vector>
#include <Eigen/StdVector>
#include <bits/stdc++.h>

using namespace std;
using namespace Eigen;

using TransformT = Matrix<double, 6, 1>;
using JacobianT = TransformT;
using HessianT = Matrix<double, 6, 6>;
using JacobianCoefficientsT = Matrix<double, 8, 3>;
using HessianCoefficientsT = Matrix<double, 18, 3>;
using PointJacobianT = Matrix<double, 3, 6>;
using PointHessianT = Matrix<double, 18, 6>;
using CovarianceMatrixT = Matrix<double, 3, 3>;
using PointT = Matrix<double, 3, 1>;
using PointTT = Matrix<double, 1, 3>;

class ScoreParams
{
public:
  ScoreParams(const double &outlier_prob, const double &resolution)
  {
    vector<double> c(2);
    d_ = vector<double>(3);

    // calc c1, c2 in (eq.6.8) [Magnusson 2009]
    c[0] = 10.0 * (1 - outlier_prob);
    c[1] = outlier_prob / pow(resolution, 3);

    d_[2] = -log(c[1]);
    d_[0] = -log(c[0] + c[1]) - d_[2];
    d_[1] = -2 * log((-log(c[0] * exp(-0.5) + c[1]) - d_[2]) / d_[0]);
  }

  double d(const unsigned int i) const{
    return d_[i - 1];
  }
private:
  std::vector<double> d_;
};





tuple<double, JacobianT, HessianT> computeDerivative
(const ScoreParams &param,
 const PointJacobianT &point_jacobian, const PointHessianT &point_hessian,
 const PointT &x_k_dash, const CovarianceMatrixT &cov_inv){

  PointTT xkd_T_conv_inv =  x_k_dash.transpose() * cov_inv;
  double xkd_T_conv_inv_xkd = xkd_T_conv_inv * x_k_dash;
  double exp_term = exp(-param.d(2) / 2 * xkd_T_conv_inv_xkd);
  double d1_d2 = param.d(1) * param.d(2);

  // calc jacobian.
  JacobianT jacobian = xkd_T_conv_inv * point_jacobian;
  jacobian *= d1_d2 * exp_term;

  // calc hessian.
  HessianT hessian;

  for(unsigned int j = 0; j < 6; ++j){
    for(unsigned int i = 0; i < 6; ++i){

      double t1 = xkd_T_conv_inv * point_jacobian.col(i);
      double t2 = xkd_T_conv_inv * point_jacobian.col(j);
      double t3 = xkd_T_conv_inv * point_hessian.block<3, 1>(i * 3, 0);
      double t4 = point_jacobian.col(j).transpose() * cov_inv * point_jacobian.col(i);

      hessian(i, j) = d1_d2 * exp_term *
        (-param.d(2) *
         t1 * t2 + t3 + t4);
    }
  }


  // calc score.
  double score = -param.d(1) * exp_term;

  return make_tuple(score, jacobian, hessian);
}

void AngleDerivatives(TransformT & tf,JacobianCoefficientsT & jacobian_coefficients,HessianCoefficientsT & hessian_coefficients){
    double cx, cy, cz, sx, sy, sz;
    if (fabs(tf(3)) < 10e-5) {
        cx = 1.0;
        sx = 0.0;
    } else {
        cx = cos(tf(3));
        sx = sin(tf(3));
    }
    if (fabs(tf(4)) < 10e-5) {
        cy = 1.0;
        sy = 0.0;
    } else {
        cy = cos(tf(4));
        sy = sin(tf(4));
    }
    if (fabs(tf(5)) < 10e-5) {
        cz = 1.0;
        sz = 0.0;
    } else {
        cz = cos(tf(5));
        sz = sin(tf(5));
    }
    jacobian_coefficients <<    (-sx * sz + cx * sy * cz),  (-sx * cz - cx * sy * sz),  (-cx * cy),
                                (cx * sz + sx * sy * cz),   (cx * cz - sx * sy * sz),   (-sx * cy),
                                (-sy * cz),                 sy * sz,                    cy,
                                sx * cy * cz,               (-sx * cy * sz),            sx * sy,
                                (-cx * cy * cz),            cx * cy * sz,               (-cx * sy),
                                (-cy * sz),                 (-cy * cz),                 0,
                                (cx * cz - sx * sy * sz),   (-cx * sz - sx * sy * cz),  0,
                                (sx * cz + cx * sy * sz),   (cx * sy * cz - sx * sz),   0;
    
    hessian_coefficients <<     0,                          0,                          0,
                                (-cx * sz - sx * sy * cz),  (-cx * cz + sx * sy * sz),  sx * cy,
                                (-sx * sz + cx * sy * cz),  (-cx * sy * sz - sx * cz),  (-cx * cy),
                                0,                          0,                          0,
                                (cx * cy * cz),             (-cx * cy * sz),            (cx * sy),
                                (sx * cy * cz),             (-sx * cy * sz),            (sx * sy),
                                0,                          0,                          0,
                                (-sx * cz - cx * sy * sz),  (sx * sz - cx * sy * cz),   0,
                                (cx * cz - sx * sy * sz),   (-sx * sy * cz - cx * sz),  0,
                                (-cy * cz),                 (cy * sz),                  (sy),
                                (-sx * sy * cz),            (sx * sy * sz),             (sx * cy),
                                (cx * sy * cz),             (-cx * sy * sz),            (-cx * cy),
                                (sy * sz),                  (sy * cz),                  0,
                                (-sx * cy * sz),            (-sx * cy * cz),            0,
                                (cx * cy * sz),             (cx * cy * cz),             0,
                                (-cy * cz),                 (cy * sz),                  0,
                                (-cx * sz - sx * sy * cz),  (-cx * cz + sx * sy * sz),  0,
                                (-sx * sz + cx * sy * cz),  (-cx * sy * sz - sx * cz),  0;
}

void PointDerivatives(PointT & point,JacobianCoefficientsT & jacobian_coefficients,HessianCoefficientsT & hessian_coefficients,PointJacobianT & point_jacobian,PointHessianT & point_hessian){
    double jacobian_params[8]={0,0,0,0,0,0,0,0};
    vector<Vector3d,aligned_allocator<Vector3d> > hessian_params(6);
    for(int i=0;i<8;i++){
        for(int j=0;j<3;j++){
            jacobian_params[i] += point(j) * jacobian_coefficients(i,j);
        }
    }
    for(int i=0;i<6;i++){
        hessian_params[i] << 0,0,0;
        for(int j=0;j<3;j++){
            hessian_params[i](0) += point(j) * hessian_coefficients(i*3+0,j);
            hessian_params[i](1) += point(j) * hessian_coefficients(i*3+1,j);
            hessian_params[i](2) += point(j) * hessian_coefficients(i*3+2,j);
        }
    }
    point_jacobian = MatrixXd::Zero(3, 6);
    point_hessian = MatrixXd::Zero(18, 6);
    point_jacobian(0,0) = 1;
    point_jacobian(1,1) = 1;
    point_jacobian(2,2) = 1;
    point_jacobian(1, 3) = jacobian_params[0];
    point_jacobian(2, 3) = jacobian_params[1];
    point_jacobian(0, 4) = jacobian_params[2];
    point_jacobian(1, 4) = jacobian_params[3];
    point_jacobian(2, 4) = jacobian_params[4];
    point_jacobian(0, 5) = jacobian_params[5];
    point_jacobian(1, 5) = jacobian_params[6];
    point_jacobian(2, 5) = jacobian_params[7];
    point_hessian.block<3, 1>(9, 3) =  hessian_params[0];
    point_hessian.block<3, 1>(12, 3) = hessian_params[1];
    point_hessian.block<3, 1>(15, 3) = hessian_params[2];
    point_hessian.block<3, 1>(9, 4) =  hessian_params[1];
    point_hessian.block<3, 1>(12, 4) = hessian_params[3];
    point_hessian.block<3, 1>(15, 4) = hessian_params[4];
    point_hessian.block<3, 1>(9, 5) =  hessian_params[2];
    point_hessian.block<3, 1>(12, 5) = hessian_params[4];
    point_hessian.block<3, 1>(15, 5) = hessian_params[5];

}

#endif