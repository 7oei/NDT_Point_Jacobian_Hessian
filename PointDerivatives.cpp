#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
#include <math.h>
#include <vector>
#include <Eigen/StdVector>

void AngleDerivatives(Eigen::Matrix<float, 6, 1> & tf,Eigen::Matrix<float, 8, 3> & jacobian_coefficients,Eigen::Matrix<float, 18, 3> & hessian_coefficients){
    float cx, cy, cz, sx, sy, sz;
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

void PointDerivatives(Eigen::Matrix<float, 3, 1> & point,Eigen::Matrix<float, 8, 3> & jacobian_coefficients,Eigen::Matrix<float, 18, 3> & hessian_coefficients,Eigen::Matrix<float, 3, 6> & jacobian,Eigen::Matrix<float, 18, 6> & hessian){
    float jacobian_params[8]={0,0,0,0,0,0,0,0};
    std::vector<Eigen::Vector3f,Eigen::aligned_allocator<Eigen::Vector3f> > hessian_params(6);
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
    jacobian = Eigen::MatrixXf::Zero(3, 6);
    hessian = Eigen::MatrixXf::Zero(18, 6);
    jacobian(0,0) = 1;
    jacobian(1,1) = 1;
    jacobian(2,2) = 1;
    jacobian(1, 3) = jacobian_params[0];
    jacobian(2, 3) = jacobian_params[1];
    jacobian(0, 4) = jacobian_params[2];
    jacobian(1, 4) = jacobian_params[3];
    jacobian(2, 4) = jacobian_params[4];
    jacobian(0, 5) = jacobian_params[5];
    jacobian(1, 5) = jacobian_params[6];
    jacobian(2, 5) = jacobian_params[7];
    hessian.block<3, 1>(9, 3) =  hessian_params[0];
    hessian.block<3, 1>(12, 3) = hessian_params[1];
    hessian.block<3, 1>(15, 3) = hessian_params[2];
    hessian.block<3, 1>(9, 4) =  hessian_params[1];
    hessian.block<3, 1>(12, 4) = hessian_params[3];
    hessian.block<3, 1>(15, 4) = hessian_params[4];
    hessian.block<3, 1>(9, 5) =  hessian_params[2];
    hessian.block<3, 1>(12, 5) = hessian_params[4];
    hessian.block<3, 1>(15, 5) = hessian_params[5];

}

int main(){

    Eigen::Matrix<float, 6, 1> tf;
    tf << 0,0,0,0,0,M_PI;
    Eigen::Matrix<float, 3, 1> point;
    point << 1,0,0;
    Eigen::Vector3f mean;
    mean << 0,0,0;
    Eigen::MatrixXf cov = Eigen::MatrixXf::Identity(3, 3);
    // std::cout << tf << std::endl;
    // std::cout << std::endl;
    // std::cout << cov << std::endl;
    // std::cout << std::endl;

    Eigen::Matrix<float, 8, 3> jacobian_coefficients;
    Eigen::Matrix<float, 18, 3> hessian_coefficients;
    AngleDerivatives(tf,jacobian_coefficients,hessian_coefficients);
    std::cout << "jacobian_coefficients" << std::endl;
    std::cout << jacobian_coefficients << std::endl;
    std::cout << std::endl;
    std::cout << "hessian_coefficients" << std::endl;
    std::cout << hessian_coefficients << std::endl;
    std::cout << std::endl;

    Eigen::Matrix<float, 3, 6> jacobian;
    Eigen::Matrix<float, 18, 6> hessian;
    PointDerivatives(point,jacobian_coefficients,hessian_coefficients,jacobian,hessian);
    std::cout << "jacobian" << std::endl;
    std::cout << jacobian << std::endl;
    std::cout << std::endl;
    std::cout << "hessian" << std::endl;
    std::cout << hessian << std::endl;
    std::cout << std::endl;
    return 0;

}