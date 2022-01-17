#include "PointDerivatives.hpp"

using namespace std;

int main(){

    TransformT tf;
    tf << 0,0,0,0,0,M_PI;
    PointT point;
    point << 1,0,0;
    PointT mean;
    mean << 0,0,0;
    CovarianceMatrixT cov = MatrixXd::Identity(3, 3);
    CovarianceMatrixT cov_inv = MatrixXd::Identity(3, 3);
    cov_inv = cov.inverse();
    // cout << tf << endl;
    // cout << endl;
    // cout << cov << endl;
    // cout << endl;

    JacobianCoefficientsT jacobian_coefficients;
    HessianCoefficientsT hessian_coefficients;
    AngleDerivatives(tf,jacobian_coefficients,hessian_coefficients);
    cout << "Jacobian Coefficients is " << endl;
    cout << jacobian_coefficients << endl;
    cout << endl;
    cout << "Hessian Coefficients is " << endl;
    cout << hessian_coefficients << endl;
    cout << endl;

    PointJacobianT point_jacobian;
    PointHessianT point_hessian;
    PointDerivatives(point,jacobian_coefficients,hessian_coefficients,point_jacobian,point_hessian);
    cout << "Point Jacobian is " << endl;
    cout << point_jacobian << endl;
    cout << endl;
    cout << "Point Hessian is " << endl;
    cout << point_hessian << endl;
    cout << endl;

    double output_prob = 0.055;
    double resolution = 1.0;
    ScoreParams params(output_prob, resolution);
    cout << "d Params is " << params.d(1)<< "," << params.d(2) << "," << params.d(3) << endl;

    PointT x_k_dash;
    x_k_dash = point - mean;

    tuple<double, JacobianT, HessianT> derivatives = computeDerivative(params, point_jacobian, point_hessian, x_k_dash, cov_inv);
    cout << "Score is "<< get<0>(derivatives) << endl;
    cout << endl;
    cout << "Jacobian is " << endl;
    cout << get<1>(derivatives) << endl;
    cout << endl;
    cout << "Hessian is " << endl;
    cout << get<2>(derivatives) << endl;
    cout << endl;

    cout << "All End" << endl;


}