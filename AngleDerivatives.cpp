#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
#include <math.h>

void AngleDerivatives(Eigen::Matrix<double, 6, 1> & tf,Eigen::Matrix<double, 8, 3> & jacobian_coefficients,Eigen::Matrix<double, 18, 3> & hessian_coefficients){
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

int main(){

    Eigen::Matrix<double, 6, 1> tf;
    tf << 0,0,0,0,0,M_PI;
    Eigen::Vector3f point;
    point << 1,0,0;
    Eigen::Vector3f mean;
    mean << 0,0,0;
    Eigen::MatrixXf cov = Eigen::MatrixXf::Identity(3, 3);

    std::cout << tf << std::endl;
    std::cout << std::endl;
    std::cout << cov << std::endl;
    std::cout << std::endl;
    Eigen::Matrix<double, 8, 3> jacobian_coefficients;
    Eigen::Matrix<double, 18, 3> hessian_coefficients;
    AngleDerivatives(tf,jacobian_coefficients,hessian_coefficients);
    std::cout << jacobian_coefficients << std::endl;
    std::cout << std::endl;
    std::cout << hessian_coefficients << std::endl;
}